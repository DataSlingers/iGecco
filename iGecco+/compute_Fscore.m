function [Fscore1,Fscore2,Fscore3,Fscore_overall] = compute_Fscore(active_set,oracle_set1,oracle_set2,oracle_set3,p1,p2,p3)

oracle_null_set1 = setdiff(1:p1,oracle_set1);
oracle_null_set2 = setdiff((p1+1):(p1+p2),oracle_set2);
oracle_null_set3 = setdiff((p1+p2+1):(p1+p2+p3),oracle_set3);

active_set1 = intersect(active_set,1:p1);
active_set2 = intersect(active_set,(p1+1):(p1+p2));
active_set3 = intersect(active_set,(p1+p2+1):(p1+p2+p3));


% FP/FN rate
null_set1 = setdiff(1:p1,active_set1);
false_positives1 = length(intersect(active_set1,oracle_null_set1));
false_negatives1 = length(intersect(oracle_set1,null_set1));
FP1 = false_positives1 /max(length(oracle_null_set1),1);
FN1 = false_negatives1 /length(oracle_set1);

Recall = 1 - FN1;
Precision = 1 - false_positives1/length(active_set1);
Fscore1 = 2 * Recall * Precision / (Recall + Precision);



null_set2 = setdiff((p1+1):(p1+p2),active_set2);
false_positives2 = length(intersect(active_set2,oracle_null_set2));
false_negatives2 = length(intersect(oracle_set2,null_set2));
FP2 = false_positives2 /max(length(oracle_null_set2),1);
FN2 = false_negatives2 /length(oracle_set2);

Recall = 1 - FN2;
Precision = 1 - false_positives2/length(active_set2);
Fscore2 = 2 * Recall * Precision / (Recall + Precision);


null_set3 = setdiff((p1+p2+1):(p1+p2+p3),active_set3);
false_positives3 = length(intersect(active_set3,oracle_null_set3));
false_negatives3 = length(intersect(oracle_set3,null_set3));
FP3 = false_positives3 /max(length(oracle_null_set3),1);
FN3 = false_negatives3 /length(oracle_set3);

Recall = 1 - FN3;
Precision = 1 - false_positives3/length(active_set3);
Fscore3 = 2 * Recall * Precision / (Recall + Precision);


oracle_null_set = [oracle_null_set1 oracle_null_set2 oracle_null_set3];
null_set = [null_set1 null_set2 null_set3];
oracle_set = [oracle_set1 oracle_set2 oracle_set3];
null_set_overall = setdiff(1:(p1+p2+p3),active_set);
false_positives_overall = length(intersect(active_set,oracle_null_set));
false_negatives_overall = length(intersect(oracle_set,null_set));
FP_overall = false_positives_overall /max(length(oracle_null_set),1);
FN_overall = false_negatives_overall /length(oracle_set);


Recall = 1 - FN_overall;
Precision = 1 - false_positives_overall/length(active_set);
Fscore_overall = 2 * Recall * Precision / (Recall + Precision);











end

