package weka.classifiers.meta;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.RandomizableClassifier;
import weka.core.*;
import weka.filters.AllFilter;
import weka.filters.Filter;

import java.io.Serializable;
import java.util.*;

public class FilterTree extends RandomizableClassifier implements Serializable {
    protected class TreeNode implements Serializable{
        // References to related TreeNodes
        protected TreeNode leftBranch;
        protected TreeNode rightBranch;
        protected TreeNode parentNode;

        // Values that the tree node needs to remember
        protected Attribute attribute;
        protected double splitPoint;
        protected double info;
        protected int[] predictedCounts = null;
        protected double[] predictedProbabilities = null;
        protected Filter localFilter;

        public TreeNode(Instances instances, TreeNode parent) throws Exception{
            this.parentNode = parent;

            // Creates a new local filter
            this.localFilter = Filter.makeCopy(m_Filter);
            if(this.localFilter instanceof Randomizable){
                ((Randomizable)this.localFilter).setSeed(m_random.nextInt());
            }
            this.localFilter.setInputFormat(instances);

            Instances filteredInstances = Filter.useFilter(instances, this.localFilter);
            Object[] result = FindBestSplitPointForAllAttributes(filteredInstances);

            // If there was no attribute to split on
            if(result == null){
                createLeafNode(instances);
            }
            else{
                this.attribute = (Attribute)result[0];
                this.splitPoint= (double)result[1];
                this.info = (double)result[2];

                // If the stop critera are met
                if(instances.size() <= getMinimumNumberOfInstancesToStop() || (this.parentNode != null && this.parentNode.info - this.info == 0)){
                    createLeafNode(instances);
                }
                else{
                    Instances[] newInstances = distributeInstancesAcrossBranches(filteredInstances, instances);
                    Instances leftInstances = newInstances[0];
                    Instances rightInstances = newInstances[1];

                    this.leftBranch = new TreeNode(leftInstances, this);
                    this.rightBranch = new TreeNode(rightInstances, this);
                }
            }
        }

        // Creates a new leaf node with the given instances
        protected void createLeafNode(Instances instances){
            this.predictedCounts = new int[instances.numClasses()];
            this.predictedProbabilities = new double[instances.numClasses()];

            // Adds up the counts of each class values
            for(int i = 0; i < instances.size(); i++){
                this.predictedCounts[(int)instances.instance(i).classValue()] ++;
            }

            // Converts the counts into probabilites
            for(int i = 0; i < this.predictedProbabilities.length; i++){
                this.predictedProbabilities[i] = this.predictedCounts[i] / (double)instances.size();
            }
        }

        // Distributes the instances based on the split point
        protected Instances[] distributeInstancesAcrossBranches(Instances filteredInstances, Instances instances){
            // We have the best attribute to split on now
            Instances leftInstances = new Instances(instances, 0);
            Instances rightInstances = new Instances(instances, 0);

            for(int i = 0; i < instances.size(); i++){
                // Puts all values less than or equal to the split value in the left branch
                if(filteredInstances.instance(i).value(this.attribute) <= this.splitPoint){
                    leftInstances.add(instances.instance(i));
                }
                else{
                    rightInstances.add(instances.instance(i));
                }
            }

            return new Instances[] {leftInstances, rightInstances};
        }

        // Finds the best split point for all attributes
        protected Object[] FindBestSplitPointForAllAttributes(Instances filteredInstances) throws Exception{
            Attribute bestAttribute = null;
            double bestSplitPointValue = 0;
            double bestInfo = 0;

            int numAttributes = filteredInstances.numAttributes();
            int classIndex = filteredInstances.classIndex();
            // Finds the best split point for each attribute
            for(int i = 0; i < numAttributes; i++){
                if(i != classIndex){
                    double[] result = FindBestSplitPointForGivenAttribute(filteredInstances.attribute(i), filteredInstances);
                    if(result != null){
                        double splitPointValue = result[0];
                        double splitPointIndex = result[1];
                        double informationGain = result[2];

                        // If the new information gain value is better
                        if(bestAttribute == null || bestInfo > informationGain){
                            bestAttribute = filteredInstances.attribute(i);
                            bestSplitPointValue = splitPointValue;
                            bestInfo = informationGain;
                        }
                    }
                }
            }

            if(bestAttribute == null){
                return null;
            }
            else{
                return new Object[] {bestAttribute, bestSplitPointValue, bestInfo};
            }
        }

        // Finds the best split point for a given attribute
        protected double[] FindBestSplitPointForGivenAttribute(Attribute currentAttribute, Instances instances){
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

            // Checks each possible split point
            for(int i = 0; i < sortedInstances.size() - 1; i++){
                // Recalculate class counts
                double classValue = sortedInstances.instance(i).classValue();
                leftClassCount[(int)classValue]++;
                rightClassCount[(int)classValue]--;

                // if the two adjacent values are identical
                if(sortedInstances.instance(i).value(currentAttribute) != sortedInstances.instance(i+1).value(currentAttribute)){
                    double infoGain = informationGain(leftClassCount, rightClassCount);
                    double newSplitPoint = (sortedInstances.instance(i).value(currentAttribute) + sortedInstances.instance(i+1).value(currentAttribute)) / 2;
                    // If the new best info should be changed
                    if((bestSplitPointIndex == -1 || infoGain < bestInformationGain) && newSplitPoint != sortedInstances.instance(i).value(currentAttribute)
                            && newSplitPoint != sortedInstances.instance(i+1).value(currentAttribute)){
                        bestInformationGain = infoGain;
                        bestSplitPointValue = newSplitPoint;
                        bestSplitPointIndex = i;
                    }
                }
            }

            if(bestSplitPointIndex == -1){
                return null;
            }
            else{
                return new double[] {bestSplitPointValue, bestSplitPointIndex, bestInformationGain};
            }

        }

        // Sorts by a given attribute
        protected Instances SortByAttribute(Attribute currentAttribute, Instances instances){
            Instances oldInstances =  new Instances(instances, 0, instances.size());
            Instances newInstances = new Instances(oldInstances, 0);
            int iterations = oldInstances.size();
            // Finds the smallest value of the instance at the given attribute, and adds it to the new instancces
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

        // classifies a given instance
        protected double[] classify(Instance instance) throws Exception{
            // If the current node is a non leaf node
            if(this.predictedProbabilities == null){
                // Filter the instance
                this.localFilter.input(instance);
                Instance filteredInstance = this.localFilter.output();

                // Decide which split branch to go down
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

        @Override
        public String toString() {
            return this.toStringLevel(0);
        }

        protected String toStringLevel(int level){
            String spaceString = spaceString(level);
            // If the current node is a splitter node
            if(this.predictedProbabilities == null){
                return "\n" + spaceString + this.attribute.name() + " <= " + round(this.splitPoint, m_numDecimalPlaces)
                        + this.leftBranch.toStringLevel(level + 1)+ "\n"
                        + spaceString + this.attribute.name() + " > " + round(this.splitPoint, m_numDecimalPlaces)
                        + this.rightBranch.toStringLevel(level + 1);
            }
            else{
                return ": " + countArrayString();
            }
        }

        // Turns the count arrays into a string
        private String countArrayString(){
            String returnString = "";
            for(int i = 0; i < this.predictedCounts.length; i++){
                returnString += this.predictedCounts[i] + " ";
            }
            return returnString;
        }

        // Creates the padding string for the toString
        private String spaceString(int level){
            String returnString = "";
            String spaceCharacter = "|  ";
            for(int i = 0; i < level; i++){
                returnString += spaceCharacter;
            }
            return returnString;
        }

        // Rounds the number value to the given number of decimal places
        private double round(double value, int places) {
            if (places < 0) throw new IllegalArgumentException();

            long factor = (long) Math.pow(10, places);
            value = value * factor;
            long tmp = Math.round(value);
            return (double) tmp / factor;
        }
    }

    public static final long serialVersionUID = 6583114962L;
    protected TreeNode filterTree;
    protected Random m_random;

    protected int m_minimumNumberOfInstancesToStop = 1;
    @OptionMetadata(
            displayName = "minNumInstancesForLeafNode",
            description = "Minimum number of instances for a leaf node", displayOrder = 1,
            commandLineParamName = "M",
            commandLineParamSynopsis = "-M <>")
    public void setMinimumNumberOfInstancesToStop(int min){ m_minimumNumberOfInstancesToStop = min;}
    public int getMinimumNumberOfInstancesToStop(){return m_minimumNumberOfInstancesToStop;}

    // Sets the filter to use at each node of the tree

    protected Filter m_Filter = new AllFilter();
    @OptionMetadata(
            displayName = "Filter",
            description = "The filter to use", displayOrder = 2,
            commandLineParamName = "F",
            commandLineParamSynopsis = "-F <filter specification>")
    public void setFilter(Filter filter){m_Filter = filter;};
    public Filter getFilter(){return m_Filter;}

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        m_random = instances.getRandomNumberGenerator(getSeed());
        filterTree = new TreeNode(instances, null);
    }

    @Override
    public double[] distributionForInstance(Instance var1) throws Exception {
        return filterTree.classify(var1);
    }

    // Calculates the information on a binary split
    protected double informationGain(int[] left, int[] right){
        // Calculate the sums of the splits
        int leftSum = 0;
        for(int val: left){
            leftSum += val;
        }
        int rightSum = 0;
        for(int val: right){
            rightSum += val;
        }
        double total = leftSum + rightSum;

        // Calculate the entropy of each split
        double leftEntropy = entropy(left, leftSum);
        double rightEntropy = entropy(right, rightSum);
        return ((leftSum / total) * leftEntropy) + ((rightSum / total) * rightEntropy);
    }

    // Calculates the entropy of a set of values
    protected double entropy(int[] values, int total){
        double result = 0;
        for(int val: values){
            double dVal = (double)val;
            result += - (dVal/total) * logBase2(dVal/total);
        }
        return result;
    }

    // calculaes the log base 2 of a given value
    protected double logBase2(double value){
        // defines log(0) as 0
        if(value == 0.0){
            return 0.0;
        }
        else{
            return Math.log(value)/Math.log(2);
        }
    }

    @Override
    public String toString(){
        return this.filterTree.toString();
    }
}
