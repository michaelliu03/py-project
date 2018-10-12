import fasttext

lrs = [0.01, 0.05, 0.002]
dims = [5, 10, 25, 50, 75, 100]

best_tr, best_val = 0, 0
for lr in lrs:
    for dim in dims:
        classifier = fasttext.supervised(input_file='data/intent_small_train.txt',
                                         output='data/intent_model',
                                         label_prefix='__label__',
                                         dim=dim,
                                         lr=lr,
                                         epoch=50)
        result_tr = classifier.test('data/intent_small_train.txt')
        result_val = classifier.test('data/intent_small_test.txt')

        if result_tr.precision > best_tr:
            best_tr = result_tr.precision
            params_tr = (lr, dim, result_tr)

        if result_val.precision > best_val:
            best_val = result_val.precision
            params_val = (lr, dim, result_val)

print(best_tr)
print(params_tr)