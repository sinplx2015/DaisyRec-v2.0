import os
import pandas as pd
import json
import argparse

def load_data(dataset):
    if dataset == 'ml-1m':
        movies_path = os.path.join('data', dataset, 'movies.dat')
        ratings_path = os.path.join('data', dataset, 'ratings.dat')

        movies = pd.read_csv(movies_path, sep='::', header=None, names=['MovieID', 'Title', 'Genres'], engine='python', encoding='ISO-8859-1')
        ratings = pd.read_csv(ratings_path, sep='::', header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'], engine='python', encoding='ISO-8859-1') # not utf-8

    elif dataset == 'ml-100k':
        raise ValueError("Not yet")

    else:
        raise ValueError("Unsupported dataset. Please use 'ml-1m' or 'ml-100k'.")

    return movies, ratings

def extract_genre_mapping(movies):
    genre_set = set()
    for genres in movies['Genres']:
        genre_set.update(genres.split('|'))
    genre_list = ['Unknown'] + sorted(genre_set)  
    return {genre: idx for idx, genre in enumerate(genre_list)}

def create_item_to_category_mapping(movies, genre_to_id):
    item_to_category = {}
    for _, row in movies.iterrows():
        movie_id = row['MovieID']
        genres = row['Genres'].split('|')
        if genres == ['']: 
            genre_ids = [genre_to_id['Unknown']]
        else:
            genre_ids = [genre_to_id[genre] for genre in genres]
        item_to_category[movie_id] = genre_ids
    return item_to_category

def save_mapping(mapping, filename):
    with open(filename, 'w') as f:
        json.dump(mapping, f)

def main(dataset):
    movies, _ = load_data(dataset)
    genre_to_id = extract_genre_mapping(movies)
    item_to_category = create_item_to_category_mapping(movies, genre_to_id)

    output_dir = os.path.join('data', dataset)
    os.makedirs(output_dir, exist_ok=True)

    save_mapping(genre_to_id, os.path.join(output_dir, 'genre_to_id.json'))
    save_mapping(item_to_category, os.path.join(output_dir, 'item_to_category.json'))

    print(f"Genre to ID mapping and item to category mapping saved in {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate category mappings for a given dataset.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., ml-1m, ml-100k)')
    args = parser.parse_args()

    main(args.dataset)
