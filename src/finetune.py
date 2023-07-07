import hydra
from omegaconf import DictConfig, OmegaConf
@hydra.main(version_base=None, config_path="configs", config_name="config")
def my_app(cfg : DictConfig) -> None:
    # Set the output directory to the project root

    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    my_app()