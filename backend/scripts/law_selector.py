import json
import argparse
import tqdm
from pathlib import Path


def load_maliyat_data(json_path):
    """Load maliyat.json file and return the result list."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('result', [])


def select_laws_by_lawid(laws, lawid):
    """
    Select laws with lawid from the laws list.
    
    Args:
        laws: List of law dictionaries from maliyat.json
        lawid: Optional specific lawid to filter by. If None, returns all laws with lawid field.
    
    Returns:
        List of laws containing lawid field
    """
    with tqdm.tqdm(total=len(laws), desc='Filtering laws by lawId') as pbar:
        laws_with_id = []
        for law in laws:
            if 'lawId' in law:
                if lawid is None or law['lawId'] == lawid:
                    laws_with_id.append(law)
            pbar.update(1)
    
    with tqdm.tqdm(total=len(laws_with_id), desc='Filtering laws by specific lawId') as pbar:
        filtered_laws = []
        for law in laws_with_id:
            if law['lawId'] == lawid:
                filtered_laws.append(law)
            pbar.update(1)
    
    return filtered_laws


def get_unique_lawids(laws):
    """Get list of unique lawIds from laws."""
    return sorted(list(set(law['lawId'] for law in laws if 'lawId' in law)))


def main():
    parser = argparse.ArgumentParser(description='Select laws by lawId from maliyat.json')
    parser.add_argument('--law-id', type=int, required=True, help='The lawId to filter by')
    parser.add_argument('--src', type=Path, required=True, help='Path to maliyat.json file')
    parser.add_argument('--dst', type=Path, required=True, help='Path to save selected laws')
    args = parser.parse_args()
    
    # Paths
    source_path = args.src
    output_path = args.dst

    # Load data
    laws = load_maliyat_data(str(source_path))
    
    # Select laws with specific lawid
    selected_laws = select_laws_by_lawid(laws, lawid=args.law_id)
    
    if selected_laws:
        print(f"Found {len(selected_laws)} law(s) with lawId {args.law_id}")
        json_output = json.dumps(selected_laws, ensure_ascii=False, indent=2)
        if output_path: 
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_output)
            print(f"Selected laws saved to {output_path}")
    else:
        print(f"No laws found with lawId {args.law_id}")


if __name__ == '__main__':
    main()
