package weka.classifiers.trees;

import org.w3c.dom.Attr;
import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.AllFilter;
import weka.filters.Filter;

import java.util.*;

public class FilterTree extends AbstractClassifier {

    private class TreeNode{
        public TreeNode leftBranch;
        public TreeNode rightBranch;
        public TreeNode parentNode;

        public Attribute attribute;
        public double splitPoint;
        public double info;

        public Instances instances;

        public TreeNode(Instances instances){
            for(int i = 0; i < instances.numAttributes(); i++){

            }
        }

        public double[] FindBestSplitPointForAttribute(Attribute attribute, Instances instances){
            int numAttributes = instances.numAttributes();

            for(int i = 0; i < numAttributes; i++){
                Instances newInstances = new Instances(instances, instances.size());
                Attribute currentAttribute = instances.attribute(i);
                SortByAttribute(currentAttribute, instances, newInstances);

                double[] bestSplitPoint = FindBestSplitPoint(currentAttribute, newInstances);

                // Stopping criteria
                if(parentNode != null && (parentNode.info - bestSplitPoint[2] == 0)){
                    // TODO: STOP
                }
                else{
                    return bestSplitPoint;
                }

            }
        }

        private double[] FindBestSplitPoint(Attribute currentAttribute, Instances instances){
            Map<Double, Integer> leftMap = new HashMap<Double, Integer>();
            Map<Double, Integer> rightMap = new HashMap<Double, Integer>();

            double bestSplitPoint = 0;
            double bestInformationGain = 0;
            int bestSplitPointIndex = -1;

            for(int i = 0; i < instances.size(); i++){
                double classValue = instances.instance(i).classValue();
                if(leftMap.containsKey(classValue)){
                    rightMap.put(classValue, (rightMap.get(classValue) + 1));
                }
                else{
                    leftMap.put(classValue, 0);
                    rightMap.put(classValue, 1);
                }
            }

            for(int i = 0; i < instances.size(); i++){
                // Recalculate class counts
                double classValue = instances.instance(0).classValue();
                rightMap.put(classValue, (rightMap.get(classValue) -1));
                leftMap.put(classValue, (leftMap.get(classValue) + 1));

                if(instances.instance(i).value(currentAttribute) != instances.instance(i+1).value(currentAttribute)){
                    Integer[] leftMapArray = (Integer[])leftMap.values().toArray();
                    Integer[] rightMapArray = (Integer[])rightMap.values().toArray();

                    double infoGain = informationGain(leftMapArray, rightMapArray);

                    if(bestSplitPointIndex == -1 || infoGain < bestInformationGain){
                        bestInformationGain = infoGain;
                        bestSplitPoint = (instances.instance(i).value(currentAttribute) + instances.instance(i+1).value(currentAttribute)) / 2;
                        bestSplitPointIndex = i;
                    }
                }
            }

            return new double[] {bestSplitPoint, bestSplitPointIndex, bestInformationGain};
        }

        // Sorts by a given attribute
        private void SortByAttribute(Attribute currentAttribute, Instances oldInstances, Instances newInstances){
            int iterations = oldInstances.size();
            for(int i = 0; i < iterations; i++){
                Double smallestValue = Double.NaN;
                Instance smallestInstance = null;
                int smallestInstanceIndex = -1;

                for(int j = 0; j < oldInstances.size(); j++){
                    double value = oldInstances.instance(j).value(currentAttribute);

                    if(smallestValue.isNaN() || value < smallestValue){
                        smallestValue = value;
                        smallestInstance = oldInstances.instance(j);
                        smallestInstanceIndex = j;
                    }
                }

                oldInstances.remove(smallestInstanceIndex);
                newInstances.add(smallestInstance);
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

    }

    public double[] distributionForInstance(Instances instances) throws Exception {
        return null;
    }

    // Calculates the information gain on a binary split
    public double informationGain(Integer[] left, Integer[] right){
        int leftSum = 0;
        for(int val: left){
            leftSum += val;
        }
        int rightSum = 0;
        for(int val: right){
            rightSum += val;
        }
        int total = leftSum + rightSum;

        return ((leftSum / total) * entropy(left, leftSum)) + ((rightSum / total) * entropy(right, rightSum));
    }

    // Calculates the
    public double entropy(Integer[] values, Integer total){
        int result = 0;
        for(int val: values){
            result += - (val/total) * logBase2(val/total);
        }
        return result;
    }

    public double logBase2(double value){
        return Math.log(value)/Math.log(2);
    }
}
