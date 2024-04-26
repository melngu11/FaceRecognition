

%% Load AT&T Face Data 
imds = imageDatastore('data\', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames', ...
    'ReadFcn', @customReadFcn);
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, 'randomized');

% Display some sample images
numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages, 16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I);
end

%% Load Pretrained Network

net = vgg19;
analyzeNetwork(net);

inputSize = net.Layers(1).InputSize;

%% Replace Final Layers

%Layers before fc8 (first 22) kept for transfer learning
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(imdsTrain.Labels));

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
    softmaxLayer
    classificationLayer];

%% Train Network

pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',4, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

netTransfer = trainNetwork(augimdsTrain,layers,options);
%% Classify Validation Images
[YPred,scores] = classify(netTransfer,augimdsValidation);

idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end

YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation)
%% ------ Functions -------

function I = customReadFcn(filename)
    I = imread(filename);
    if size(I, 3) == 1
        I = repmat(I, [1, 1, 3]);
    end
end


