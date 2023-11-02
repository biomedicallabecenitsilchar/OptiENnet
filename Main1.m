clc;           
clear all;
close all;
gpuDevice(1);
p=0.7;
matlabpath=('E:\sonal project\optimized_paper');
data=fullfile(matlabpath,'TrainData1');
traindata=imageDatastore(data,'IncludeSubfolders',true,'LabelSource','foldernames');
tbl = countEachLabel(traindata);

minSetCount = min(tbl{:,2});
[imdsTrain,imdsValidation]= splitEachLabel(traindata,p,'randomize');

% resize the images to match the network input layers
augimdsTrain = augmentedImageDatastore([224 224 3],imdsTrain,'ColorPreprocessing', 'gray2rgb');
augimdsTest = augmentedImageDatastore([224 224 3],imdsValidation,'ColorPreprocessing', 'gray2rgb');

net = resnet50;
lgraph=layerGraph(net);
newLearnableLayer = fullyConnectedLayer(2, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
    % Replacing the last layers with new layers
    
    lgraph = replaceLayer(lgraph,'fc1000',newLearnableLayer);
    newsoftmaxLayer = softmaxLayer('Name','new_softmax');
    lgraph = replaceLayer(lgraph,'fc1000_softmax',newsoftmaxLayer);
    newClassLayer = classificationLayer('Name','new_classoutput');
    lgraph = replaceLayer(lgraph,'ClassificationLayer_fc1000',newClassLayer);
    analyzeNetwork(lgraph)

Problem.obj = @Sphere;
Problem.nVar = 20;

M = 20; % number of chromosomes (cadinate solutions)
N = Problem.nVar;  % number of genes (variables)
MaxGen = 1000;
Pc = 0.85;
Pm = 0.01;
Er = 0.05;
visualization = 1; % set to 0 if you do not want the convergence curve 

[BestChrom]  = GeneticAlgorithm (M , N, MaxGen , Pc, Pm , Er , Problem.obj , visualization)

disp('The best chromosome found: ')
BestChrom.Gene
disp('The best fitness value: ')
BestChrom.Fitness
batchsize=4;
K2 = 0.0001
K3 = 0.01

     options = trainingOptions('sgdm', ...
    'InitialLearnRate',K2, ...
    'MaxEpochs',BestChrom.Fitness, ...
    'L2Regularization', K3,...    
    'Shuffle','every-epoch', ...
    'MiniBatchSize', batchsize,...
    'ValidationData',augimdsTest , ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'ExecutionEnvironment','auto');   
    
    net = trainNetwork(augimdsTrain,lgraph,options);
save net net

featureLayer = 'new_fc';
trainingFeatures = activations(net, augimdsTrain, featureLayer, ...
      'OutputAs', 'columns');
    
    trainingFeatures = double(trainingFeatures);
    testLabels = imdsValidation.Labels;
    % Get training labels from the trainingSet
    trainingLabels = imdsTrain.Labels;
    
    testFeatures = activations(net, augimdsTest, featureLayer, ...
        'MiniBatchSize', 64, 'OutputAs', 'columns');
    
    testFeatures = double(testFeatures);
    fprintf('Creating the target matrix for tarining of ANN\n');
    
    dimTrain = size(imdsTrain.Files,1);
    dimTest = size(imdsValidation.Files,1);

 no_person=2;
 no_img_p_s_train = 630;
    TrainTargets = zeros(no_person, dimTrain);
    for j = 1:no_person
        for k = 1:no_img_p_s_train
            TrainTargets(j,((j-1)*no_img_p_s_train + k)) = 1;
        end
    end
    fprintf('Saving on disk TrainTargets \n'); save TrainTargets  TrainTargets ;
    % %
    fprintf('Creating the target matrix of TestData for calculating accuracy(Performance)\n');
    no_img_p_s_test = 900-no_img_p_s_train;
    
    TestTargets = zeros(no_person, dimTest);
    for j = 1:no_person
        for k = 1:no_img_p_s_test
            TestTargets(j,((j-1)*no_img_p_s_test + k)) = 1;
        end
    end
    fprintf('Saving on disk TestTargets \n'); save TestTargets TestTargets;
    
    TrainTargets_ind=vec2ind(TrainTargets);
    TestTargets_ind=vec2ind(TestTargets);

    ELM_Train=[TrainTargets_ind' trainingFeatures'];
    ELM_Test=[TestTargets_ind' testFeatures'];
    
    N=800;
    option.N=N;
    option.ActivationFunction='my_fuzzy';
    option.C=2^4;
    option.Scale=1;
    option.Scalemode=1;
    option.bias=0;
    option.link=0;
 [TrT, TtT, TrainAcc, TestAcc] = My_ELM_Old(ELM_Train, ELM_Test,1, 20000, 'my_fuzzy');
 [train_accuracy,test_accuracy]=my_RVFL(trainingFeatures',TrainTargets_ind',testFeatures',TestTargets_ind',option); 
 [TestAcc] = ELM_Test1(ELM_Train, ELM_Test, 1, 20000, 'my_fuzzy')
    
 
  
