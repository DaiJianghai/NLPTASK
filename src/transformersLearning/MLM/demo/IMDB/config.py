class Config:
    """
    Class to store the configuration of the model
    """
    def __init__(self):
        # file path 
        self.data_path = r"D:\DevelopmentProgress\Project_VSCode\deeplearning\src\transformersLearning\MLM\data\IMDB\IMDB Dataset.csv"
        self.small = r"D:\DevelopmentProgress\Project_VSCode\deeplearning\src\transformersLearning\MLM\data\IMDB\small.csv"
        self.model_path = r"E:\Huggingface_Model\BERT\bert-base-uncased"
        self.max_len = 128
        # training config
        self.train_batch_size = 4
        self.epochs = 100
        self.learning_rate = 5e-5
        # valid config
        self.valid_batch_size = 4
        # test config
        self.test_batch_size = 4