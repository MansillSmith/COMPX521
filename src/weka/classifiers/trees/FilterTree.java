package weka.classifiers.trees;

import weka.classifiers.AbstractClassifier;
import weka.core.Instances;
import weka.filters.AllFilter;
import weka.filters.Filter;

import java.util.ArrayList;

public class FilterTree extends AbstractClassifier {

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

    public double InformationGain(ArrayList<Integer> values, int total){

    }

    public double entropy(int value1, int value2){
        int sum = value1 + value2;
        return -(value1/sum) * logBase2(value1/sum) - (value2/sum) * logBase2(value2/sum);
    }

    public double logBase2(double value){
        return Math.log(value)/Math.log(2);
    }
}
