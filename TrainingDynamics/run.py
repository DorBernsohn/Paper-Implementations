from getData import getData
for dataset_name in ['rte', 'qqp', 'mrpc', 'cola', 'qnli']:
    model_name = 'distilbert-base-cased'
    train_dynamic = getData(dataset_name=dataset_name, model_name=model_name, max_length=128, learning_rate=3e-5, epochs=10)
    train_dynamic.load_data()
    train_dynamic.train_model()
    train_dynamic.kde_plot()