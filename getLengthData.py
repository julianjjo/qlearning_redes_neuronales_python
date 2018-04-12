import pickle

def main():
    with open('output_data_positive', 'rb') as fp:
        output_data_positive = pickle.load(fp)
    with open('output_data_negative', 'rb') as fp:
        output_data_negative = pickle.load(fp)
    with open('input_training', 'rb') as fp:
        input_training = pickle.load(fp)
    print("Cantidad de output_data_positive: {}".format(len(output_data_positive)))
    print("Cantidad de output_data_negative: {}".format(len(output_data_negative)))
    print("Cantidad de input_training: {}".format(len(input_training)))

    for i in range(0, 20):
        print(input_training[i])
        print(output_data_positive[i])
        print(output_data_negative[i])


if __name__ == '__main__':
    main()
