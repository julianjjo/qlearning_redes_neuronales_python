import pickle

def main():
    with open('output_data', 'rb') as fp:
        output_data = pickle.load(fp)
    with open('input_training', 'rb') as fp:
        input_training = pickle.load(fp)
    print("Cantidad de output_data: {}".format(len(output_data)))
    print("Cantidad de input_training: {}".format(len(input_training)))

    # for i in range(0, 80):
    #     print(input_training[i])
    #     print(output_data[i])
    #     print(output_data[i])


if __name__ == '__main__':
    main()
