load PredictedLabels; load ExpectedLabel; load TY;
load testY; load Yt_temp; load testY_temp;

combinedPredictions = vertcat(testY, PredictedLabels');

ensemblePredictions = majority_voting(combinedPredictions);

accuracy = sum(ensemblePredictions == TestTargets_ind) / numel(TestTargets_ind);
fprintf('Ensemble Accuracy : %.2f%%\n', accuracy * 100);

y_test =  repmat(TestTargets_ind, 1,2);

  EVAL = Evaluate(y_test', ensemblePredictions) 
  confMat = confusionmat(y_test', ensemblePredictions);
  confMat1 = bsxfun(@rdivide,confMat,sum(confMat,2));
  cm= confusionchart(confMat, {'Non RD','RD'});
  


