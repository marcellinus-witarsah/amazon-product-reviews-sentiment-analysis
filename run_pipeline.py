from pipelines.training_pipeline import training_pipeline

if __name__ == "__main__":
    training_pipeline(
        data_path="data/interim/sampled-raw-data.csv"
    )
