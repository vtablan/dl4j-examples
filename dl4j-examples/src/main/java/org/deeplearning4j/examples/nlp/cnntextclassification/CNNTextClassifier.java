package org.deeplearning4j.examples.nlp.cnntextclassification;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.PreprocessorVertex;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.ReshapePreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.StreamLineIterator;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Simple example illustrating the use of a Convolutional Neural Network to perform text classification.
 * This example is inspired by a
 * <a href="http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/">blog post</a> by
 * Denny Britz.
 *
 * @author vtablan 2016/10/31
 */
public class CNNTextClassifier {

    /**
     * A text classification instance: the text, and the class (encoded as a one-hot binary vector).
     */
    public static class Example {
        public final List<String> words;

        public final float[] classOneHot;

        public Example(List<String> words, float[] classOneHot) {
            this.words = words;
            this.classOneHot = classOneHot;
        }
    }

    private static final int MIN_WORD_FREQUENCY = 3;
    public static final String NEGATIVES = "/cnntextclassification/rt-polarity.neg.txt";
    public static final String POSITIVES = "/cnntextclassification/rt-polarity.pos.txt";
    protected static long SEED = 42;

    protected static int[] FILTER_SIZES = new int[]{1, 2, 3};

    protected static int[] FILTER_COUNTS = new int[]{128, 128, 128};

    protected static int EMBEDDING_DIMS = 100;

    protected static double DROPOUT_PROB = 05d;


    public static List<Example> loadData() throws IOException {
        TokenizerFactory tokFct = new DefaultTokenizerFactory();

        List<Example> data = new ArrayList<>();
        for (String resName : new String[]{POSITIVES, NEGATIVES}) {
            float[] classOneHot = resName == POSITIVES ? new float[]{1, 0} : new float[]{0, 1};
            try (InputStream inputStream = CNNTextClassifier.class.getResourceAsStream(resName)) {
                SentenceIterator sentIter = new StreamLineIterator.Builder(inputStream).build();
                while (sentIter.hasNext()) {
                    Tokenizer tokenizer = tokFct.create(sentIter.nextSentence());
                    data.add(new Example(tokenizer.getTokens(), classOneHot));
                }
            }
        }
        Collections.shuffle(data, new Random(SEED));
        return data;
    }

    public static ComputationGraph makeGraphModel(int iterations, int numClasses, int docLength, WordVectors embeddings) {
        String outputLayerName = "Output";
        String inputLayerName = "Input";
        ComputationGraphConfiguration.GraphBuilder graphConfBuilder = new NeuralNetConfiguration.Builder()
            .seed(SEED)
            .iterations(iterations)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(1e-2)
            .adamMeanDecay(0.9)
            .adamVarDecay(0.999)
            .updater(Updater.ADAM)
            .graphBuilder()
            .pretrain(false)
            .backprop(true)
            .addInputs(inputLayerName) // input shape is [docLength, embDim], single channel
            .setOutputs(outputLayerName);

        // input shape before embedding layer is [ docLength]
        // i.e. a row vector with one int per word
        // embedding layer expects a column vector
//        String preEmbsReshapedName = "PreEmbsReshaped";
//        graphConfBuilder.addVertex(preEmbsReshapedName, new PreprocessorVertex(
//                new ReshapePreProcessor(new int[]{1, docLength}, new int[]{docLength, 1}, false)),
//            inputLayerName);

        // embedding layer
        String embeddingLayerName = "Embedding";
        int embeddingsDim = embeddings.lookupTable().layerSize();
        EmbeddingLayer embeddingLayerConf = new EmbeddingLayer.Builder()
            .nIn(embeddings.vocab().numWords())
            .nOut(embeddingsDim)
            .activation("identity")
            .name(embeddingLayerName)
            .updater(Updater.NONE)  //fixed embeddings
            .build();
        graphConfBuilder.addLayer(embeddingLayerName, embeddingLayerConf, inputLayerName);
        // data shape now: [docLength, embDim]
        // CNN layer expects an image, i.e. a 4 order tensor of shape [batchSize, channels, height, width]
        String reshaperName = "PreConvReshaped";
        graphConfBuilder.addVertex(reshaperName, new PreprocessorVertex(
                new ReshapePreProcessor(new int[]{docLength, embeddingsDim}, new int[]{1, 1, docLength, embeddingsDim}, false)),
            embeddingLayerName);
        // data shape now: [batchSize, channels, height, width]

        List<String> cnnOutputsToMerge = new ArrayList<>();
        for (int filterPos = 0; filterPos < FILTER_SIZES.length; filterPos++) {
            int filterSize = FILTER_SIZES[filterPos];
            int filtersCount = FILTER_COUNTS[filterPos];
            // convolution
            String convLayerName = String.format("ConvFilter-%d", filterSize);
            ConvolutionLayer convolutionLayerConf = new ConvolutionLayer.Builder()
                .kernelSize(filterSize, embeddingsDim)
                .nIn(1) // single channel in
                .nOut(filtersCount) // number of filters as out
                .stride(1, 1)
                .activation("relu")
                .name(convLayerName)
                .build();
            graphConfBuilder.addLayer(convLayerName, convolutionLayerConf, reshaperName);
            // data shape now: [1, docLength - filterSize + 1, filtersCount]
            int convolvedRows = docLength - filterSize + 1;
            // add a pooling layer after the convolution
            String poolLayerName = convLayerName + "-pooled";
            SubsamplingLayer poolingLayerConf = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(convolvedRows, 1) // we max over all the output from the conv layer
                //                .kernelSize(-1, 1)
                .stride(1, 1)
                .name(poolLayerName)
                .build();
            graphConfBuilder.addLayer(poolLayerName, poolingLayerConf, convLayerName);
            // data shape now: [1, 1, 1, filtersCount]

            String postPoolReshaperName = poolingLayerConf + "-postReshape";
            graphConfBuilder.addVertex(postPoolReshaperName, new PreprocessorVertex(
                    new ReshapePreProcessor(new int[]{1, 1, 1, filtersCount}, new int[]{1, filtersCount}, false)),
                poolLayerName);
            // output shape: [1, filtersCount]
            cnnOutputsToMerge.add(postPoolReshaperName);
        }
        // join all pools into a single vector
        graphConfBuilder.addVertex("MergedPools", new MergeVertex(), cnnOutputsToMerge.toArray(new String[cnnOutputsToMerge.size()]));
        // output shape is [1, sum(filtersCount)]
        // dense layer
        String denseLayerName = "DenseWithDropout";
        int allFilters = 0;
        for (int filterCount : FILTER_COUNTS) {
            allFilters += filterCount;
        }
        DenseLayer denseLayerConf = new DenseLayer.Builder()
            .nIn(allFilters)
            .nOut(numClasses)
            .dropOut(DROPOUT_PROB)
            .name(denseLayerName)
            .build();
        graphConfBuilder.addLayer(denseLayerName, denseLayerConf, "MergedPools");
        // data shape now: [1, numClasses]
        // output Softmax
        OutputLayer outputLayerConf = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .nIn(numClasses)
            .nOut(numClasses)
            .activation("softmax")
            .name(outputLayerName)
            .build();
        graphConfBuilder.addLayer(outputLayerName, outputLayerConf, denseLayerName);
        // output shape: [1, numClasses]

        ComputationGraph computationGraph = new ComputationGraph(graphConfBuilder.build());
        computationGraph.init();

        // initialise the embeddings layer
        org.deeplearning4j.nn.layers.feedforward.embedding.EmbeddingLayer embeddingsLayer =
            (org.deeplearning4j.nn.layers.feedforward.embedding.EmbeddingLayer) computationGraph.getLayer(embeddingLayerName);
        embeddingsLayer.setParam(DefaultParamInitializer.WEIGHT_KEY, embeddings.lookupTable().getWeights());
        return computationGraph;
    }

    public static void main(String[] args) throws Exception {
        WordVectors embeddings = WordVectorSerializer.loadTxtVectors(new File(args[0]));
        List<Example> data = loadData();
        int maxDocLength = 0;
        for (Example example : data) {
            if (example.words.size() > maxDocLength) maxDocLength = example.words.size();
        }
        ComputationGraph computationGraph = makeGraphModel(100, 2, maxDocLength, embeddings);

        int split = (int) Math.floor(data.size() * 0.8);
        List<Example> trainingData = data.subList(0, split);
        List<Example> testData = data.subList(split, data.size());
        AtomicInteger globalStep = new AtomicInteger(0);
        for (Example example : trainingData) {
            System.out.println("Step " + globalStep.getAndIncrement());
            computationGraph.fit(asDataset(example, embeddings, maxDocLength, 2));
        }
    }

    private static DataSet asDataset(Example example, WordVectors embeddings, int maxDocLength, int numClasses) {
        float[] featuresData = new float[maxDocLength];
        float[] labelsData = new float[numClasses];

        for (int wordIdx = 0; wordIdx < example.words.size(); wordIdx++) {
            String word = example.words.get(wordIdx);
            if (!embeddings.hasWord(word)) {
                word = embeddings.getUNK();
            }
            featuresData[wordIdx] = embeddings.indexOf(word);
        }

        System.arraycopy(example.classOneHot, 0, labelsData, 0, numClasses);
        return new org.nd4j.linalg.dataset.DataSet(Nd4j.create(featuresData, new int[]{maxDocLength, 1}), Nd4j.create(labelsData));
    }
}
