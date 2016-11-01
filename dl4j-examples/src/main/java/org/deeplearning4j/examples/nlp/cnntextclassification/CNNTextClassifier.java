package org.deeplearning4j.examples.nlp.cnntextclassification;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.PreprocessorVertex;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.ReshapePreProcessor;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.StreamLineIterator;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
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

    protected static class InertPreprocessorVertexConf extends PreprocessorVertex {
        private final InputPreProcessor preProcessor;

        public InertPreprocessorVertexConf(InputPreProcessor preProcessor) {
            super(preProcessor);
            this.preProcessor = preProcessor;
        }

        @Override
        public GraphVertex instantiate(ComputationGraph graph, String name, int idx, INDArray paramsView, boolean initializeParams) {
            return new org.deeplearning4j.nn.graph.vertex.impl.PreprocessorVertex(graph, name, idx, preProcessor) {
                @Override
                public Pair<Gradient, INDArray[]> doBackward(boolean tbptt) {
                    return new Pair<>(null, null);
                }
            };
        }
    }

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

    public static final String NEGATIVES = "/cnntextclassification/rt-polarity.neg.txt";
    public static final String POSITIVES = "/cnntextclassification/rt-polarity.pos.txt";
    protected static long SEED = 42;

    protected static int BATCH_SIZE = 16;

    protected static int[] FILTER_SIZES = new int[]{1, 2, 3};

    protected static int[] FILTER_COUNTS = new int[]{128, 128, 128};

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

    /**
     * Performs text classification by applying the following layers:
     * <table>
     * <tr>
     * <th>Layer</th>
     * <th>Input Shape</th>
     * <th>Output Shape</th>
     * <th>Purpose</th>
     * </tr>
     * <tr>
     * <td>Reshape</td>
     * <td>[batchSize, docLength]</td>
     * <td>[batchSize * docLength, 1]</td>
     * <td>Prepare input for embedding lookup</td>
     * </tr>
     * <tr>
     * <td>Embedding</td>
     * <td>[batchSize * docLength, 1]</td>
     * <td>[batchSize * docLength, embDim]</td>
     * <td>Convert word IDs into dense word vectors</td>
     * </tr>
     * <tr>
     * <td>Reshape</td>
     * <td>[batchSize * docLength, embDim]</td>
     * <td>[batchSize, 1, docLength, embDim]</td>
     * <td>Turn embedded word sequence into 1-channel rectangular input for convolutions</td>
     * </tr>
     * <tr>
     * <td>Convolution - filterSize (in parallel, for each filterSize)</td>
     * <td>[batchSize, 1, docLength, embDim]</td>
     * <td>[batchSize, filterCount, docLength - filterSize + 1, 1]</td>
     * <td>Apply convolutions, to detect relevant phrases.</td>
     * </tr>
     * <tr>
     * <td>Max pooling - filterSize (in parallel, for each filterSize)</td>
     * <td>[batchSize, filterCount, docLength - filterSize + 1, 1]</td>
     * <td>[batchSize, filterCount, 1, 1]</td>
     * <td>Reduce dimensionality by max-pooling-over-time, applied over the whole document sequence</td>
     * </tr>
     * <tr>
     * <td>Reshape - filterSize (in parallel, for each filterSize)</td>
     * <td>[batchSize, filterCount, 1, 1]</td>
     * <td>[batchSize, filterCount]</td>
     * <td>Reshape data to prepare it as input to the dense layer</td>
     * </tr>
     * <tr>
     * <td>MergeVertex (joins all the pools, over all filter sizes)</td>
     * <td>[batchSize, filterCount], for each filter size</td>
     * <td>[batchSize, sum(filterCount)]</td>
     * <td>Concatenate the output from all the multiple filters of different sizes</td>
     * </tr>
     * <tr>
     * <td>Dense layer</td>
     * <td>[batchSize, sum(filterCount)]</td>
     * <td>[batchSize, numClasses]</td>
     * <td>Perform classification, with drop-out</td>
     * </tr>
     * <tr>
     * <td>Output (Softmax) layer (produce class probabilities as output)</td>
     * <td>[batchSize, sum(filterCount)]</td>
     * <td>[batchSize, numClasses]</td>
     * <td>Convert logits from the dense layer into output class probabilities; generate the loss function to be optimised during training</td>
     * </tr>
     * </table>
     * <p>
     * Where:
     * <dl>
     * <dt>batchSize</dt>
     * <dd>the number of documents in a mini-batch</dd>
     * <dt>doclen</dt>
     * <dd>the number of tokens in a document</dd>
     * <dt>embDim</dt>
     * <dd>the word embedding dimension</dd>
     * <dt>filterSize</dt>
     * <dd>the size of one convolutional filter</dd>
     * <dt>filterCount</dt>
     * <dd>the number of convolutional filters of a given size</dd>
     * <dt>numClasses</dt>
     * <dd>the number of output classes</dd>
     * </dl>
     *
     * @param iterations
     * @param numClasses
     * @param docLength
     * @param embeddings
     * @return
     */
    public static ComputationGraph makeGraphModel(int batchSize, int iterations, int docLength, int numClasses, WordVectors embeddings) {
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
            .addInputs(inputLayerName) // input shape is [batchSize, docLength]
            .setOutputs(outputLayerName); // output shape is [batchSize, numClasses]

        // input shape before embedding layer is [batchSize, docLength]
        // embedding layer expects a column vector of [exampleCount, 1], where each example is 1 word.
        String reshapedForEmbeddingName = "ReshapedForEmbedding";

        graphConfBuilder.addVertex(reshapedForEmbeddingName,
            new InertPreprocessorVertexConf(new ReshapePreProcessor(new int[]{batchSize, docLength}, new int[]{batchSize * docLength, 1}, false)),
//            new PreprocessorVertex(new ReshapePreProcessor(new int[]{batchSize, docLength}, new int[]{batchSize * docLength, 1}, false)),
            inputLayerName);

        // embedding layer
        String embeddedName = "Embedded";
        int embeddingsDim = embeddings.lookupTable().layerSize();
        EmbeddingLayer embeddingLayerConf = new EmbeddingLayer.Builder()
            .nIn(embeddings.vocab().numWords())
            .nOut(embeddingsDim)
            .activation("identity")
            .name(embeddedName)
            .updater(Updater.NONE)  //fixed embeddings
            .build();
        graphConfBuilder.addLayer(embeddedName, embeddingLayerConf, reshapedForEmbeddingName);
        // data shape now: [batchSize * docLength, embDim]

        // CNN layer expects an image, i.e. a 4 order tensor of shape [batchSize, channels, height, width]
        String reshapedForConv = "ReshapedForConv";
        graphConfBuilder.addVertex(reshapedForConv, new PreprocessorVertex(
            new ReshapePreProcessor(new int[]{batchSize * docLength, embeddingsDim}, new int[]{batchSize, 1, docLength, embeddingsDim}, false)
        ), embeddedName);
        // data shape now: [batchSize, 1, doclength, embsDim], interpreted as ~[batchSize, channels, height, width]

        // we will apply a series of convolutions, after which we'll concatenate the outputs
        // here we store the names of the outputs to concatenate
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
            graphConfBuilder.addLayer(convLayerName, convolutionLayerConf, reshapedForConv);
            int convolvedRows = docLength - filterSize + 1;
            // data shape now: [batchSize, filterCount, convolvedRows, 1]

            // add a max-pooling layer after the convolution
            String poolLayerName = convLayerName + "-pooled";
            SubsamplingLayer poolingLayerConf = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(convolvedRows, 1) // we max over all the output from the conv layer
                .stride(1, 1)
                .name(poolLayerName)
                .build();
            graphConfBuilder.addLayer(poolLayerName, poolingLayerConf, convLayerName);
            // data shape now:  [batchSize, filtersCount, 1, 1]

            // Reshape data to prepare for dense layers
            String postPoolReshaperName = poolLayerName + "-postReshaped";

            graphConfBuilder.addVertex(postPoolReshaperName, new PreprocessorVertex(
                new CnnToFeedForwardPreProcessor(1, 1, filtersCount)
                //new ReshapePreProcessor(new int[]{batchSize, filtersCount, 1, 1}, new int[]{batchSize, filtersCount}, false)
            ), poolLayerName);
            // data shape now: [batchSize, filtersCount]
            cnnOutputsToMerge.add(postPoolReshaperName);
        }
        // join all pools into a single vector
        String mergedPoolsName = "MergedPools";
        graphConfBuilder.addVertex(mergedPoolsName, new MergeVertex(), cnnOutputsToMerge.toArray(new String[cnnOutputsToMerge.size()]));
        //data shape now: [batchSize, sum(filtersCount)]

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
        graphConfBuilder.addLayer(denseLayerName, denseLayerConf, mergedPoolsName);
        // data shape now: [batchSize, numClasses]
        // output Softmax
        OutputLayer outputLayerConf = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .nIn(numClasses)
            .nOut(numClasses)
            .activation("softmax")
            .name(outputLayerName)
            .build();
        graphConfBuilder.addLayer(outputLayerName, outputLayerConf, denseLayerName);
        // output shape: [batchSize, numClasses]

        ComputationGraph computationGraph = new ComputationGraph(graphConfBuilder.build());
        computationGraph.init();

        // initialise the embeddings layer
        org.deeplearning4j.nn.layers.feedforward.embedding.EmbeddingLayer embeddingsLayer =
            (org.deeplearning4j.nn.layers.feedforward.embedding.EmbeddingLayer) computationGraph.getLayer(embeddedName);

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
        ComputationGraph computationGraph = makeGraphModel(BATCH_SIZE, 100, maxDocLength, 2, embeddings);

        int split = (int) Math.floor(data.size() * 0.8);
        // round up to multiple of batch size
        split += split % BATCH_SIZE;
        List<Example> trainingData = data.subList(0, split);
        List<Example> testData = data.subList(split, data.size());
        AtomicInteger globalStep = new AtomicInteger(0);
        while (globalStep.get() < trainingData.size() / BATCH_SIZE) {
            System.out.println("Step " + globalStep.get());
            int batchStart = globalStep.getAndIncrement() * BATCH_SIZE;
            List<Example> batch = trainingData.subList(batchStart, batchStart + BATCH_SIZE);
            computationGraph.fit(asDataset(batch, embeddings, maxDocLength, 2));
        }


    }

    private static DataSet asDataset(List<Example> batch, WordVectors embeddings, int maxDocLength, int numClasses) {
        float[][] featuresData = new float[batch.size()][maxDocLength];
        float[][] labelsData = new float[batch.size()][numClasses];

        for (int exampleIdx = 0; exampleIdx < batch.size(); exampleIdx++) {
            Example example = batch.get(exampleIdx);
            for (int wordIdx = 0; wordIdx < example.words.size(); wordIdx++) {
                String word = example.words.get(wordIdx);
                if (!embeddings.hasWord(word)) {
                    word = embeddings.getUNK();
                }
                featuresData[exampleIdx][wordIdx] = embeddings.indexOf(word);
            }
            System.arraycopy(example.classOneHot, 0, labelsData[exampleIdx], 0, numClasses);
        }
        return new org.nd4j.linalg.dataset.DataSet(Nd4j.create(featuresData), Nd4j.create(labelsData));
    }
}
