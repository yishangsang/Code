/**
 * Created by hadoop on 16-11-1.
 */
import java.io.*;
import java.text.*;
import java.util.*;

import weka.core.*;
import weka.classifiers.*;
import weka.classifiers.trees.*;
public class Cotraining_Active {
    /** Random Forest */
    protected Classifier[] m_classifiers = null;

    /** The number component */
    protected int m_numClassifiers = 15;

    /** The random seed */
    protected int m_seed = 1;

    /** Number of features to consider in random feature selection.
     If less than 1 will use int(logM+1) ) */
    protected int m_numFeatures = 0;

    /** Final number of features that were considered in last build. */
    protected int m_KValue = 0;

    private int m_numOriginalLabeledInsts = 0;

    protected double high_threshold = 0.85;



    /**
     * The constructor
     */
    public Cotraining_Active()
    {
    }

    /**
     * Set the seed for initiating the random object used inside this class
     *
     * @param s int -- The seed
     */
    public void setSeed(int s)
    {
        m_seed = s;
    }

    public void setNumClassifiers(int n)
    {
        m_numClassifiers = n;
    }

    /**
     * Set the number of features to use in random selection.
     *
     * @param n int -- Value to assign to m_numFeatures.
     */
    public void setNumFeatures(int n)
    {
        m_numFeatures = n;
    }

    //产生有放回地抽样
    public final Instances resampleWithWeights(Instances data,
                                               Random random,
                                               boolean[] sampled)
    {

        double[] weights = new double[data.numInstances()];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = data.instance(i).weight();
            //System.out.println("权重"+weights[0]);
        }
        Instances newData = new Instances(data, data.numInstances());
        if (data.numInstances() == 0) {
            return newData;
        }
        double[] probabilities = new double[data.numInstances()];
        double sumProbs = 0, sumOfWeights = Utils.sum(weights);//计算数组的总和
        for (int i = 0; i < data.numInstances(); i++) {
            sumProbs += random.nextDouble();
            probabilities[i] = sumProbs;
        }
        Utils.normalize(probabilities, sumProbs / sumOfWeights);//对概率数组按照权重总和（data.numinstances×1）进行标准化
        //System.out.println("probablities "+probabilities[3]);

        // Make sure that rounding errors don't mess things up
        probabilities[data.numInstances() - 1] = sumOfWeights;
        int k = 0; int l = 0;
        sumProbs = 0;
        while ((k < data.numInstances() && (l < data.numInstances()))) {
            if (weights[l] < 0) {
                throw new IllegalArgumentException("Weights have to be positive.");
            }
            sumProbs += weights[l];
            while ((k < data.numInstances()) &&
                    (probabilities[k] <= sumProbs)) {
                newData.add(data.instance(l));
                sampled[l] = true;
                newData.instance(k).setWeight(1);
                k++;

            }
            l++;
        }
        return newData;
    }

    public double buildClassifier(Instances labeled) throws Exception
    {
        double noise_rate = 0.0;
        boolean[][] inbags = new boolean[m_numClassifiers][];
        Random rand = new Random(m_seed);
        m_numOriginalLabeledInsts = labeled.numInstances();

        RandomTree rTree = new RandomTree();

        // set up the random tree options
        m_KValue = m_numFeatures;
        if (m_KValue < 1) m_KValue = (int) Utils.log2(labeled.numAttributes())+1;
        rTree.setKValue(m_KValue);

        m_classifiers = Classifier.makeCopies(rTree, m_numClassifiers);
        Instances[] labeleds = new Instances[m_numClassifiers];
        int[] randSeeds = new int[m_numClassifiers];
        for(int i = 0; i < m_numClassifiers; i++)
        {
            randSeeds[i] = rand.nextInt();
            ((RandomTree)m_classifiers[i]).setSeed(randSeeds[i]);
            inbags[i] = new boolean[labeled.numInstances()];
            labeleds[i] = resampleWithWeights(labeled, rand, inbags[i]);
            m_classifiers[i].buildClassifier(labeleds[i]);
        }
        noise_rate = measureNoiseRate(labeled,inbags);
        return  noise_rate;
    }
    public double classifyInstance(Instance inst) throws Exception
    {
        double[] distr = distributionForInstance(inst);
        return Utils.maxIndex(distr);
    }

    public double[] distributionForInstance(Instance inst) throws Exception
    {
        double[] res = new double[inst.numClasses()];
        for(int i = 0; i < m_classifiers.length; i++)
        {
            double[] distr = m_classifiers[i].distributionForInstance(inst);
            for(int j = 0; j < res.length; j++)
                res[j] += distr[j];
        }
        Utils.normalize(res);
        return res;//叠加每个分类器在同一类别下的概率，返回实例在每个类别下的分类概率
    }

    private double getConfidence(Instance inst) throws Exception
    {
        double[] distr = new double[inst.numClasses()];
        for(int i = 0; i < m_numClassifiers; i++)
        {
            double[] d = m_classifiers[i].distributionForInstance(inst);
            for(int iClass = 0; iClass < inst.numClasses(); iClass++)
                distr[iClass] += d[iClass];
        }
        Utils.normalize(distr);//根据数组的总和标准化
        int maxIndex = Utils.maxIndex(distr);
        inst.setClassValue(maxIndex);
        return distr[maxIndex];
    }
    //计算有放回抽样中未被抽到的样本大于置信度的错误率
    private double measureNoiseRate(Instances data, boolean[][] inbags) throws Exception
    {
        double err = 0;
        double count = 0;
        double confidence = 0;
        double fal = 0;
        for(int i = 0; i < data.numInstances() && i < m_numOriginalLabeledInsts; i++)
        {
            Instance inst = data.instance(i);
            double[] distr = new double[inst.numClasses()];
            for(int k = 0; k < m_numClassifiers; k++)
            {
                if(inbags[k][i] == true)
                    continue;
               fal++;
                double[] d = m_classifiers[k].distributionForInstance(inst);
                for(int iClass = 0; iClass < inst.numClasses(); iClass++)
                    distr[iClass] += d[iClass];
            }

            if(Utils.sum(distr) != 0)
            {
                Utils.normalize(distr);
                int maxIndex = Utils.maxIndex(distr);
                confidence = distr[maxIndex];
                //System.out.println("confidence: "+confidence);
            }

            if(confidence > high_threshold)
            {
                count ++;
                if(Utils.maxIndex(distr) != inst.classValue())
                    err ++;
            }
        }
        System.out.println("fal: "+fal);
        System.out.println("count: "+count);
        System.out.println("err: "+err);
        err /= count;
        return err;
    }

    public static void main(String[] args)
    {
        try
        {
            int seed = 0;
            int numFeatures = 0;
            Random rand = new Random(seed);
            final int NUM_CLASSIFIERS = 15;
            double confidence_value = 0.0;
            double noise_rate = 0.0;
            /** confidence threshold */
            double high_threshold = 0.85;
            double low_threshold = 0.5;

            String input_path = "G:\\实验室\\半监督 代码数据\\semi_train\\train_1\\";
            String result_path = "G:\\实验室\\半监督 代码数据\\semi_train\\result\\";
            BufferedWriter highConfidence = new BufferedWriter(new FileWriter(result_path+"highConfidence.txt"));
            BufferedWriter middleConfidence = new BufferedWriter(new FileWriter(result_path+"middleConfidence.txt"));
            BufferedWriter lowConfidence = new BufferedWriter(new FileWriter(result_path+"lowConfidence.txt"));

            BufferedReader r = new BufferedReader(new FileReader(input_path+"labeled_6.arff"));
            Instances labeled = new Instances(r);
            labeled.setClassIndex(labeled.numAttributes()-1);
            r.close();

            Cotraining_Active coactive = new Cotraining_Active();
            coactive.setNumClassifiers(NUM_CLASSIFIERS);
            coactive.setNumFeatures(numFeatures);
            coactive.setSeed(rand.nextInt());
            noise_rate = coactive.buildClassifier(labeled);
            System.out.println("高置信度样本的噪声率为： "+noise_rate);

            //分类器的准确率
            r = new BufferedReader(new FileReader(input_path+"test.arff"));
            Instances test = new Instances(r);
            test.setClassIndex(labeled.numAttributes()-1);
            r.close();

            double err = 0;
            double acc = 0;
            for(int i = 0; i < test.numInstances(); i++)
            {
                if(coactive.classifyInstance(test.instance(i)) != test.instance(i).classValue())
                    err++;
            }
            acc = 1 - err/test.numInstances();
            System.out.println("协同分类器的正确率为： " + acc);

            //按照置信度对标记结果进行排序
            r = new BufferedReader(new FileReader(input_path+"unlabeled_6.arff"));
            Instances unlabeled = new Instances(r);
            int sum = unlabeled.numInstances();
            TreeMap<Double,String> treeMap = new TreeMap<Double,String>();
            unlabeled.setClassIndex(labeled.numAttributes()-1);
            r.close();
            BufferedReader Reader = new BufferedReader(new FileReader(input_path+"unlabeled_6.arff"));
            String lineTxt = "";
            while((lineTxt = Reader.readLine())!=null){
                if((lineTxt.indexOf("@")<0)&&(!lineTxt.equals(""))){
                    break;
                }
            }

            List<String> list = new ArrayList<String>();
            int high_count = 0;
            for(int i=0;i<sum;i++)
            {
                confidence_value = coactive.getConfidence(unlabeled.instance(i));
                if(confidence_value>=high_threshold)
                    high_count++;
                int indexClass = lineTxt.indexOf("%");
                lineTxt = lineTxt.substring(indexClass+1);
                for(int j=0;j<i;j++)
                {
                    if(list.size()==0)
                        list.add(0,unlabeled.instance(i).toString() + " %" + lineTxt+ " %"+confidence_value);
                    else
                    {
                        String[] value = list.get(j).split(" %");
                        double element = Double.valueOf(value[value.length-1]);
                        if(confidence_value<=element) {
                            list.add(j, unlabeled.instance(i).toString() + " %" + lineTxt + " %" + confidence_value);
                            break;
                        }
                    }
                }
                if(list.size()==i)
                    list.add(i, unlabeled.instance(i).toString() + " %" + lineTxt + " %" + confidence_value);
                lineTxt = Reader.readLine();
                //System.out.println(list.size());
            }
            System.out.println("高于阈值的样本数量："+high_count);
            System.out.println("标记样本数量："+list.size());

            //对生成的标记样本按照置信度排列顺序进行分类

            for(int i=0;i<173;i++)
            {
                lowConfidence.write(list.get(i).split(" %")[0]+" %"+list.get(i).split(" %")[1]);
                lowConfidence.newLine();
            }
            for(int i=173;i<list.size()-170;i++)
            {
                middleConfidence.write(list.get(i).split(" %")[0]+" %"+list.get(i).split(" %")[1]);
                middleConfidence.newLine();
            }
            for(int i=list.size()-170;i<list.size();i++)
            {
                highConfidence.write(list.get(i).split(" %")[0]+" %"+list.get(i).split(" %")[1]);
                highConfidence.newLine();
            }
            lowConfidence.close();
            middleConfidence.close();
            highConfidence.close();

        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
    }
}
