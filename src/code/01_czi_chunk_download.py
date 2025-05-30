import cellxgene_census as czi
import os
import anndata
# Constants
VERSION = "latest"
OUTPUT_DIR = "../data/cellxgene_data"
OUTPUT_FILE = "covid_PBMC"
PARTITION_SIZE = 500000  # Adjust based on memory availability
QUERY = "tissue_general == 'blood' and disease == 'COVID-19'"

# Create output directory if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_covid_blood_indices():
    """Retrieve all `soma_joinid` values"""
    with czi.open_soma(census_version=VERSION) as census:
        df = census["census_data"]["homo_sapiens"].obs.read(
            value_filter=QUERY,
            column_names=["soma_joinid"]  # Only fetch IDs to save memory
        ).concat().to_pandas()

        return df["soma_joinid"].tolist()

def download_covid_blood_data(output_file, partition_size=PARTITION_SIZE):
    """
    Download data in chunks and save as H5AD.

    Parameters:
        output_file (str): Path to save the output h5ad file.
        partition_size (int): Number of cells per partition.
    """
    # Get all matching cell IDs
    soma_ids = get_covid_blood_indices()
    total_cells = len(soma_ids)
    
    print(f"Found {total_cells} cells matching the query.")

    if total_cells == 0:
        print("No data found for COVID-19 blood samples.")
        return

    with czi.open_soma(census_version=VERSION) as census:
        for i in range(0, total_cells, partition_size):
            chunk_ids = soma_ids[i: i + partition_size]
            print(f"Downloading cells {i} to {i + len(chunk_ids)}...")

            adata = czi.get_anndata(
                census=census,
                organism="Homo sapiens",
                obs_coords=chunk_ids,  # Using actual `soma_joinid` values
                X_layers=['raw', 'normalized'],
            )

            chunk_file = f"{output_file}_chunk{i // partition_size}.h5ad"
            adata.write_h5ad(chunk_file)
            print(f"Saved {adata.shape[0]} cells to {chunk_file}")

# Run download
download_covid_blood_data(os.path.join(OUTPUT_DIR, OUTPUT_FILE))