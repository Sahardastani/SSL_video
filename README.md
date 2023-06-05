# SSL_video
hello
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

### Resize Videos

* For train, val, and test repeat the following command (change the split tag each time)
    ```
    python src/datasets/preprocessing/downscale.py
    ```

### Create CSV files

*   ```
    python src/datasets/preprocessing/prepare_csv.py 
    ```

