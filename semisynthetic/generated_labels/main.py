import raw_labels
import aggregation_labels
import trainer
import test
import pandas as pd

# M_list = [32]
M_list = [2, 4, 8, 16, 32]
prob_l_list = [0.105] * 20
# prob_l_list = [0.42] * 20
all_output = pd.DataFrame()
all_output.to_csv('all_output.csv')
epoch_size = 50

for prob_l in prob_l_list:
    print(prob_l)
    for M in M_list:
        print(M)
        raw_labels.generate_raw_labels(M=M, prob_l=prob_l)
        aggregation_labels.aggregate()
        if M == M_list[0]:
            trainer.training(epoch_size=epoch_size, method_name=['true_train_prob', 'true_train_class'])
        trainer.training(epoch_size=epoch_size)
        test.test(M=M, prob_l=prob_l)
