package weka.classifiers.trees;

import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.AllFilter;
import weka.filters.Filter;

import java.util.*;

public class FilterTree extends AbstractClassifier {

    protected TreeNode filterTree;

    private int LEAFCOUNTER = 0;
    private int SPLITTERCOUNTER = 0;

    private class TreeNode{
        public TreeNode leftBranch;
        public TreeNode rightBranch;
        public TreeNode parentNode;

        public Attribute attribute;
        public double splitPoint;
        public double info;

        public Filter localFilter;

        public double[] predictedProbabilities = null;

        public TreeNode(Instances instances, TreeNode parent) throws Exception{
            this.localFilter = Filter.makeCopy(m_Filter);
            this.localFilter.setInputFormat(instances);

            this.parentNode = parent;

            Instances filteredInstances = Filter.useFilter(instances, this.localFilter);
            Object[] result = FindBestSplitPointForAllAttributes(filteredInstances);
            this.attribute = (Attribute)result[0];
            this.splitPoint= (double)result[1];
            this.info = (double)result[2];

            if(instances.size() <= m_minimumNumberOfInstancesToStop || (this.parentNode != null && this.parentNode.info - this.info == 0)){
                int[] countArray = new int[instances.numClasses()];
                this.predictedProbabilities = new double[instances.numClasses()];

                for(int i = 0; i < instances.size(); i++){
                    countArray[(int)instances.instance(i).classValue()] ++;
                }

                for(int i = 0; i < this.predictedProbabilities.length; i++){
                    this.predictedProbabilities[i] = countArray[i] / (double)instances.size();
                }

                System.out.println("l, " + LEAFCOUNTER);
                LEAFCOUNTER++;
            }
            else{

                Instances[] newInstances = distributeInstancesAcrossBranches(filteredInstances, instances);
                Instances leftInstances = newInstances[0];
                Instances rightInstances = newInstances[1];

//                System.out.println("s, " + SPLITTERCOUNTER);
//                SPLITTERCOUNTER++;

                this.leftBranch = new TreeNode(leftInstances, this);
                this.rightBranch = new TreeNode(rightInstances, this);
            }
        }

        private Instances[] distributeInstancesAcrossBranches(Instances filteredInstances, Instances instances){
            // We have the best attribute to split on now
            Instances leftInstances = new Instances(instances, 0);
            Instances rightInstances = new Instances(instances, 0);

            for(int i = 0; i < instances.size(); i++){
                if(filteredInstances.instance(i).value(this.attribute) <= this.splitPoint){
                    leftInstances.add(instances.instance(i));
                }
                else{
                    rightInstances.add(instances.instance(i));
                }
            }

            return new Instances[] {leftInstances, rightInstances};
        }

        public Object[] FindBestSplitPointForAllAttributes(Instances filteredInstances) throws Exception{
            Attribute bestAttribute = null;
            double bestSplitPointValue = 0;
            double bestInfo = 0;

            int numAttributes = filteredInstances.numAttributes();
            int classIndex = filteredInstances.classIndex();
            for(int i = 0; i < numAttributes; i++){
//                if(filteredInstances.classAttribute().equals(filteredInstances.attribute(i))){
                if(i != classIndex){
                    double[] result = FindBestSplitPointForGivenAttribute(filteredInstances.attribute(i), filteredInstances);
                    double splitPointValue = result[0];
                    double splitPointIndex = result[1];
                    double informationGain = result[2];

                    if(bestAttribute == null || bestInfo > informationGain){
                        bestAttribute = filteredInstances.attribute(i);
                        bestSplitPointValue = splitPointValue;
                        bestInfo = informationGain;
                    }
                }
            }

            return new Object[] {bestAttribute, bestSplitPointValue, bestInfo};
        }

        private double[] FindBestSplitPointForGivenAttribute(Attribute currentAttribute, Instances instances){
            Instances sortedInstances = SortByAttribute(currentAttribute, instances);
            int[] leftClassCount = new int[sortedInstances.numClasses()];
            int[] rightClassCount = new int[sortedInstances.numClasses()];

            double bestSplitPointValue = 0;
            double bestInformationGain = 0;
            int bestSplitPointIndex = -1;

            // Initialising
            for(int i = 0; i < sortedInstances.size(); i++){
                double classValue = sortedInstances.instance(i).classValue();
                rightClassCount[(int)classValue]++;
            }

            for(int i = 0; i < sortedInstances.size() - 1; i++){
                // Recalculate class counts
                double classValue = sortedInstances.instance(i).classValue();
                leftClassCount[(int)classValue]++;
                rightClassCount[(int)classValue]--;

                if(sortedInstances.instance(i).value(currentAttribute) != sortedInstances.instance(i+1).value(currentAttribute)){
                    double infoGain = informationGain(leftClassCount, rightClassCount);
                    double newSplitPoint = (sortedInstances.instance(i).value(currentAttribute) + sortedInstances.instance(i+1).value(currentAttribute)) / 2;
                    if((bestSplitPointIndex == -1 || infoGain < bestInformationGain) && newSplitPoint != sortedInstances.instance(i).value(currentAttribute)
                            && newSplitPoint != sortedInstances.instance(i+1).value(currentAttribute)){
                        bestInformationGain = infoGain;
                        bestSplitPointValue = newSplitPoint;
                        bestSplitPointIndex = i;
                    }
                }
            }

            return new double[] {bestSplitPointValue, bestSplitPointIndex, bestInformationGain};
        }

        // Sorts by a given attribute
        private Instances SortByAttribute(Attribute currentAttribute, Instances instances){
            Instances oldInstances =  new Instances(instances, 0, instances.size());
            Instances newInstances = new Instances(oldInstances, 0);
            int iterations = oldInstances.size();
            for(int i = 0; i < iterations; i++){
                Double smallestValue = Double.NaN;
                Instance smallestInstance = null;
                int smallestInstanceIndex = -1;

                for(int j = 0; j < oldInstances.size(); j++){
                    double value = oldInstances.instance(j).value(currentAttribute);

                    if(smallestInstanceIndex == -1 || value < smallestValue){
                        smallestValue = value;
                        smallestInstance = oldInstances.instance(j);
                        smallestInstanceIndex = j;
                    }
                }

                oldInstances.remove(smallestInstanceIndex);
                newInstances.add(smallestInstance);
            }
            return newInstances;
        }

        public double[] classify(Instance instance) throws Exception{
            if(this.predictedProbabilities == null){
                Instances tempInstances = new Instances((Instances)null, 1);
                tempInstances.add(instance);

                Instances filteredInstances = Filter.useFilter(tempInstances, this.localFilter);
                Instance filteredInstance = filteredInstances.instance(0);

                double instanceValueOfSplitAttribute = filteredInstance.value(this.attribute);

                if(instanceValueOfSplitAttribute <= this.splitPoint){
                    return this.leftBranch.classify(instance);
                }
                else{
                    return this.rightBranch.classify(instance);
                }
            }
            else{
                return this.predictedProbabilities;
            }
        }
    }

    // Sets the minimum number of instances required to stop growth
    protected int m_minimumNumberOfInstancesToStop = 1;
    public void setMinimumNumberOfInstancesToStop(int min){ m_minimumNumberOfInstancesToStop = min;}
    public int getMinimumNumberOfInstancesToStop(){return m_minimumNumberOfInstancesToStop;}

    // Sets the filter to use at each node of the tree
    protected Filter m_Filter = new AllFilter();
    public void setFilter(Filter filter){m_Filter = filter;};
    public Filter getFilter(){return m_Filter;}

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        filterTree = new TreeNode(instances, null);
    }

    @Override
    public double[] distributionForInstance(Instance var1) throws Exception {
        return filterTree.classify(var1);
    }

    // Calculates the information gain on a binary split
    public double informationGain(int[] left, int[] right){
        int leftSum = 0;
        for(int val: left){
            leftSum += val;
        }
        int rightSum = 0;
        for(int val: right){
            rightSum += val;
        }
        double total = leftSum + rightSum;

        double leftEntropy = entropy(left, leftSum);
        double rightEntropy = entropy(right, rightSum);
        return ((leftSum / total) * leftEntropy) + ((rightSum / total) * rightEntropy);
    }

    // Calculates the
    public double entropy(int[] values, int total){
        double result = 0;
        for(int val: values){
            double dVal = (double)val;
            result += - (dVal/total) * logBase2(dVal/total);
        }
        return result;
    }

    public double logBase2(double value){
        if(value == 0.0){
            return 0.0;
        }
        else{
            return Math.log(value)/Math.log(2);
        }
    }

    public int[] ConvertObjectArray(Object[] objectArray){
        int[] returnArray = new int[objectArray.length];
        for(int i = 0; i < objectArray.length; i++){
            returnArray[i] = (int)objectArray[i];
        }
        return returnArray;
    }
}
