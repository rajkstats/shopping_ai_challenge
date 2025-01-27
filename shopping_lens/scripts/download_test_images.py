import requests
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import os

TEST_IMAGES = {
    "sneakers": [
        "https://images.unsplash.com/photo-1549298916-b41d501d3772",
        "https://images.unsplash.com/photo-1595950653106-6c9ebd614d3a",
        "https://images.unsplash.com/photo-1600185365926-3a2ce3cdb9eb",
        "https://images.unsplash.com/photo-1551107696-a4b0c5a0d9a2",
        "https://images.unsplash.com/photo-1579338559194-a162d19bf842",
        "https://images.unsplash.com/photo-1604671801908-6f0c6a092c05",
        "https://images.unsplash.com/photo-1491553895911-0055eca6402d",
        "https://images.unsplash.com/photo-1542291026-7eec264c27ff",
        "https://images.unsplash.com/photo-1514989771522-458c9b6c035a",
        "https://images.unsplash.com/photo-1607522370275-f14206abe5d3",
        "https://images.unsplash.com/photo-1608231387042-66d1773070a5",
        "https://images.unsplash.com/photo-1597248881519-db089d3744a5",
        "https://images.unsplash.com/photo-1584735175315-9d5df23860e6",
        "https://images.unsplash.com/photo-1552346154-21d32810aba3",
        "https://images.unsplash.com/photo-1606107557195-0e29a4b5b4aa",
        "https://images.unsplash.com/photo-1605408499391-6368c628ef42",
        "https://images.unsplash.com/photo-1560769629-975ec94e6a86",
        "https://images.unsplash.com/photo-1595341888016-a392ef81b7de",
        "https://images.unsplash.com/photo-1603808033176-9d134e6f2c74",
        "https://images.unsplash.com/photo-1603808033192-082d6919d3e1"
    ],
    "formal_shoes": [
        "https://images.unsplash.com/photo-1614252369475-531eba835eb1",
        "https://images.unsplash.com/photo-1614252370352-c4c68abb2c81",
        "https://images.unsplash.com/photo-1478186011217-80e71c1c4a5a",
        "https://images.unsplash.com/photo-1531310197839-ccf54634509e",
        "https://images.unsplash.com/photo-1546215364-12f3fff5d578",
        "https://images.unsplash.com/photo-1560343090-f0409e92791a",
        "https://images.unsplash.com/photo-1614252371353-1e3c4e4b2e3a",
        "https://images.unsplash.com/photo-1614252372786-a5f9bb3b8567",
        "https://images.unsplash.com/photo-1614252374077-6c3c89e63241",
        "https://images.unsplash.com/photo-1614252375747-7b0c9d4c8dce",
        "https://images.unsplash.com/photo-1614252376753-3e0a5c7d8a45",
        "https://images.unsplash.com/photo-1614252377759-4d9b5b5b8dce",
        "https://images.unsplash.com/photo-1614252378781-4dce0b5b6c2a",
        "https://images.unsplash.com/photo-1614252379787-5b9b5b5b8dce",
        "https://images.unsplash.com/photo-1614252380793-5b9b5b5b8dce",
        "https://images.unsplash.com/photo-1614252381799-5b9b5b5b8dce",
        "https://images.unsplash.com/photo-1614252382805-5b9b5b5b8dce",
        "https://images.unsplash.com/photo-1614252383811-5b9b5b5b8dce",
        "https://images.unsplash.com/photo-1614252384817-5b9b5b5b8dce",
        "https://images.unsplash.com/photo-1614252385823-5b9b5b5b8dce"
    ],
    "sports_shoes": [
        "https://images.unsplash.com/photo-1542291026-7eec264c27ff",
        "https://images.unsplash.com/photo-1608231387042-66d1773070a5",
        "https://images.unsplash.com/photo-1606107557195-0e29a4b5b4aa",
        "https://images.unsplash.com/photo-1605408499391-6368c628ef42",
        "https://images.unsplash.com/photo-1560769629-975ec94e6a86",
        "https://images.unsplash.com/photo-1595341888016-a392ef81b7de",
        "https://images.unsplash.com/photo-1603808033176-9d134e6f2c74",
        "https://images.unsplash.com/photo-1603808033192-082d6919d3e1",
        "https://images.unsplash.com/photo-1539185441755-769473a23570",
        "https://images.unsplash.com/photo-1562183241-b937e95585b6",
        "https://images.unsplash.com/photo-1565814636199-ae8133055c1c",
        "https://images.unsplash.com/photo-1581093458791-9d42cc05b2ef",
        "https://images.unsplash.com/photo-1581093450021-4a7360e9a6b5",
        "https://images.unsplash.com/photo-1581093458485-c87cabb7b4b7",
        "https://images.unsplash.com/photo-1581093458479-6e63b9d05a4e",
        "https://images.unsplash.com/photo-1581093458473-be6cc0a0dd0e",
        "https://images.unsplash.com/photo-1581093458467-5e6b2f0a0f7a",
        "https://images.unsplash.com/photo-1581093458461-5e6b2f0a0f7b",
        "https://images.unsplash.com/photo-1581093458455-5e6b2f0a0f7c",
        "https://images.unsplash.com/photo-1581093458449-5e6b2f0a0f7d"
    ],
    "boots": [
        "https://images.unsplash.com/photo-1608256246200-53e635b5b65f",
        "https://images.unsplash.com/photo-1608256246204-42b4e6d67fb7",
        "https://images.unsplash.com/photo-1608256246208-42b4e6d67fb8",
        "https://images.unsplash.com/photo-1608256246212-42b4e6d67fb9",
        "https://images.unsplash.com/photo-1608256246216-42b4e6d67fba",
        "https://images.unsplash.com/photo-1608256246220-42b4e6d67fbb",
        "https://images.unsplash.com/photo-1608256246224-42b4e6d67fbc",
        "https://images.unsplash.com/photo-1608256246228-42b4e6d67fbd",
        "https://images.unsplash.com/photo-1608256246232-42b4e6d67fbe",
        "https://images.unsplash.com/photo-1608256246236-42b4e6d67fbf",
        "https://images.unsplash.com/photo-1608256246240-42b4e6d67fc0",
        "https://images.unsplash.com/photo-1608256246244-42b4e6d67fc1",
        "https://images.unsplash.com/photo-1608256246248-42b4e6d67fc2",
        "https://images.unsplash.com/photo-1608256246252-42b4e6d67fc3",
        "https://images.unsplash.com/photo-1608256246256-42b4e6d67fc4",
        "https://images.unsplash.com/photo-1608256246260-42b4e6d67fc5",
        "https://images.unsplash.com/photo-1608256246264-42b4e6d67fc6",
        "https://images.unsplash.com/photo-1608256246268-42b4e6d67fc7",
        "https://images.unsplash.com/photo-1608256246272-42b4e6d67fc8",
        "https://images.unsplash.com/photo-1608256246276-42b4e6d67fc9"
    ],
    "sandals": [
        "https://images.unsplash.com/photo-1603487742131-4160ec999306",
        "https://images.unsplash.com/photo-1603487742135-4160ec999307",
        "https://images.unsplash.com/photo-1603487742139-4160ec999308",
        "https://images.unsplash.com/photo-1603487742143-4160ec999309",
        "https://images.unsplash.com/photo-1603487742147-4160ec99930a",
        "https://images.unsplash.com/photo-1603487742151-4160ec99930b",
        "https://images.unsplash.com/photo-1603487742155-4160ec99930c",
        "https://images.unsplash.com/photo-1603487742159-4160ec99930d",
        "https://images.unsplash.com/photo-1603487742163-4160ec99930e",
        "https://images.unsplash.com/photo-1603487742167-4160ec99930f",
        "https://images.unsplash.com/photo-1603487742171-4160ec999310",
        "https://images.unsplash.com/photo-1603487742175-4160ec999311",
        "https://images.unsplash.com/photo-1603487742179-4160ec999312",
        "https://images.unsplash.com/photo-1603487742183-4160ec999313",
        "https://images.unsplash.com/photo-1603487742187-4160ec999314",
        "https://images.unsplash.com/photo-1603487742191-4160ec999315",
        "https://images.unsplash.com/photo-1603487742195-4160ec999316",
        "https://images.unsplash.com/photo-1603487742199-4160ec999317",
        "https://images.unsplash.com/photo-1603487742203-4160ec999318",
        "https://images.unsplash.com/photo-1603487742207-4160ec999319"
    ]
}

def download_images():
    current_dir = Path(__file__).parent.parent
    
    # Create directories for both test and train
    test_dir = current_dir / "data" / "test_images"
    train_dir = current_dir / "data" / "train_images"
    
    for directory in [test_dir, train_dir]:
        directory.mkdir(parents=True, exist_ok=True)
        # Create class subdirectories
        for category in TEST_IMAGES.keys():
            (directory / category).mkdir(exist_ok=True)

    print("\nDownloading images...")
    for category, urls in TEST_IMAGES.items():
        print(f"\nProcessing {category} images...")
        for i, url in enumerate(tqdm(urls)):
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    # Save to both test and train directories
                    for directory in [test_dir, train_dir]:
                        img_path = directory / category / f"{category}_{i+1}.jpg"
                        with open(img_path, "wb") as f:
                            f.write(response.content)
                        
                        # Verify image can be opened
                        try:
                            img = Image.open(img_path)
                            img.verify()
                        except:
                            print(f"Corrupted image removed: {img_path}")
                            os.remove(img_path)
                            continue
                            
            except Exception as e:
                print(f"Error downloading {url}: {e}")

if __name__ == "__main__":
    download_images()