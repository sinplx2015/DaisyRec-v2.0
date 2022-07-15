import time
import yaml

from daisy.utils.parser import parse_args
from daisy.utils.splitter import TestSplitter
from daisy.utils.config import init_seed, model_config
from daisy.utils.loader import RawDataReader, Preprocessor
from daisy.utils.sampler import BasicNegtiveSampler, SkipGramNegativeSampler
from daisy.utils.dataset import convert_dataloader, BasicDataset, CandidatesDataset, UAEData
from daisy.utils.utils import get_ur, convert_npy_mat, build_candidates_set, get_adj_mat, calc_ranking_results


if __name__ == '__main__':
    ''' summarize hyper-parameter part (basic yaml + args + model yaml) '''
    config = dict()
    basic_conf = yaml.load(open('./daisy/config/basic.yaml'), Loader=yaml.loader.SafeLoader)
    config.update(basic_conf)

    args = parse_args()
    algo_name = config['algo_name'] if args.algo_name is None else args.algo_name

    model_conf = yaml.load(
        open(f'./daisy/config/model/{algo_name}.yaml'), Loader=yaml.loader.SafeLoader)
    config.update(model_conf)

    args_conf = vars(args)
    config.update(args_conf)

    ''' init seed for reproducibility '''
    init_seed(config['seed'], config['reproducibility'])
    
    ''' Test Process for Metrics Exporting '''
    reader, processor = RawDataReader(config), Preprocessor(config)
    df = reader.get_data()
    df = processor.process(df)
    user_num, item_num = processor.user_num, processor.item_num

    config['user_num'] = user_num
    config['item_num'] = item_num

    ''' Train Test split '''
    splitter = TestSplitter(config)
    train_index, test_index = splitter.split(df)
    train_set, test_set = df.iloc[train_index, :].copy(), df.iloc[test_index, :].copy()

    ''' get ground truth '''
    test_ur = get_ur(test_set)
    total_train_ur = get_ur(train_set)
    config['train_ur'] = total_train_ur

    ''' build and train model '''
    s_time = time.time()
    if config['algo_name'].lower() in ['itemknn', 'puresvd', 'slim', 'mostpop']:
        model = model_config[config['algo_name']](config)
        model.fit(train_set)
    elif config['algo_name'].lower() in ['multi-vae']:
        model = model_config[config['algo_name']](config)
        # TODO
        train_dataset = UAEData(user_num, item_num, train_set, test_set)
        training_mat = convert_npy_mat(user_num, item_num, train_set)
    elif config['algo_name'].lower() in ['mf', 'fm', 'neumf', 'nfm', 'ngcf']:
        if config['algo_name'].lower() == 'ngcf':
            _, norm_adj, _ = get_adj_mat(user_num,item_num)
            config['norm_adj'] = norm_adj

        model = model_config[config['algo_name']](config)
        sampler = BasicNegtiveSampler(train_set, config)
        train_samples = sampler.sampling()
        train_dataset = BasicDataset(train_samples)
        train_loader = convert_dataloader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
        model.fit(train_loader)
    elif config['algo_name'].lower() in ['item2vec']:
        model = model_config[config['algo_name']](config)
        sampler = SkipGramNegativeSampler(train_set, config)
        train_samples = sampler.sampling()
        train_dataset = BasicDataset(train_samples)
        train_loader = convert_dataloader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
        model.fit(train_loader)
    else:
        raise NotImplementedError('Something went wrong when building and training...')
    elapsed_time = time.time() - s_time
    print(f"Finish training: {config['dataset']} {config['prepro']} {config['algo_name']} with {config['loss_type']} and {config['sample_method']} sampling, {elapsed_time:.4f}")

    ''' build candidates set '''
    print('Start Calculating Metrics...')
    test_u, test_ucands = build_candidates_set(test_ur, total_train_ur, config)

    ''' get predict result '''
    print('')
    print('Generate recommend list...')
    print('')
    test_dataset = CandidatesDataset(test_ucands)
    test_loader = convert_dataloader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
    preds = model.rank(test_loader) # np.array (u, topk)

    if config['algo_name'].lower() in ['multi-vae']:
        # TODO
        pass

    ''' calculating KPIs '''
    print('Save metric@k result to res folder...')
    result_save_path = f"./res/{config['dataset']}/{config['prepro']}/{config['test_method']}/"
    algo_prefix = f"{config['loss_type']}_{config['algo_name']}"
    common_prefix = f"with_{config['sample_ratio']}{config['sample_method']}"

    results = calc_ranking_results(test_ur, preds, test_u, config)
    results.to_csv(f'{result_save_path}{algo_prefix}_{common_prefix}_kpi_results.csv')
