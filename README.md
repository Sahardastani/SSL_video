# SSL_video

## Dataset Preparation

### Download Videos
* **Step 1:** Clone repo and enter directory
    ```
    git clone https://github.com/cvdfoundation/kinetics-dataset.git
    cd kinetics-dataset
    ```
* **Step 2:** This will create two directories, k400 and k400_targz. Tar gzips will be in k400_targz, you can delete k400_targz directory after extraction.
    ```
    bash ./k400_downloader.sh
    ```
* **Step 3:** Extract tar gzip files
    ```
    bash ./k400_extractor.sh
    ```
### Prerequisite
* Replace `DATA.PATH_TO_DATA_DIR` in [TimeSformer_divST_8x32_224.yaml](src/configs/TimeSformer_divST_8x32_224.yaml) with your data directory.

* Append your current project direcotry in [downscale.py](src/datasets/preprocessing/downscale.py), [prepare_csv.py](src/datasets/preprocessing/prepare_csv.py), and [kinetics.py](src/datasets/kinetics.py) using `sys.path.append('current/project/directory')`. Replace it with your directory.

### Resize Videos

* Although this step is optional, it can significantly accelerate the decoding process. For train, val, and test repeat the following command (change the split tag each time)
    ```
    python src/datasets/preprocessing/downscale.py
    ```
    Remember to delete the original video folders and rename each resulting folders from `split_256` to `split`.

### Create CSV files

*   ```
    python src/datasets/preprocessing/prepare_csv.py 
    ```

## Test the Dataloader
* **Train**: Make sure `do_vis = True` for visualization.
    ``` 
    dataset = Kinetics(cfg=config, mode="train", num_retries=10, get_flow=False) 
    ```
* **Val**: Make sure `do_vis = False`, since it designed for train dataloader.
    ``` 
    dataset = Kinetics(cfg=config, mode="val", num_retries=10)
    ```