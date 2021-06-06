package weka.filters.unsupervised.instance;

import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionMetadata;
import weka.filters.SimpleBatchFilter;

import java.util.ArrayList;
import java.util.Dictionary;
import java.util.HashMap;

public class KernelHerding extends SimpleBatchFilter {

    /** for serialization */
    static final long serialVersionUID = -251831442047263433L;

    /** The kernel function to use. */
    protected Kernel m_Kernel = new PolyKernel();

    /** The subsample size, percent of original set, default 100% */
    protected double m_SampleSizePercent = 100;

    /**
     * Returns the Capabilities of this filter.
     *
     * @return the capabilities of this object
     * @see Capabilities
     */
    @Override
    public Capabilities getCapabilities() {

        Capabilities result = getKernel().getCapabilities();
        result.setOwner(this);

        result.setMinimumNumberInstances(0);

        result.enableAllClasses();
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);
        result.enable(Capabilities.Capability.NO_CLASS);

        return result;
    }

    /** Handling the kernel parameter. */
    @OptionMetadata(
            displayName = "Kernel function",
            description = "The kernel function to use.", displayOrder = 1,
            commandLineParamName = "K",
            commandLineParamSynopsis = "-K <kernel specification>")
    public Kernel getKernel() {  return m_Kernel; }
    public void setKernel(Kernel value) { m_Kernel = value; }

    /** Handling the parameter setting the sample size. */
    @OptionMetadata(
            displayName = "Percentage of the training set to sample.",
            description = "The percentage of the training set to sample (between 0 and 100).", displayOrder = 3,
            commandLineParamName = "Z",
            commandLineParamSynopsis = "-Z <double>")
    public void setSampleSizePercent(double newSampleSizePercent) { m_SampleSizePercent = newSampleSizePercent; }
    public double getSampleSizePercent() { return m_SampleSizePercent; }

    @Override
    public String globalInfo() { return "A filter implementing kernel herding for unsupervised subsampling of data."; }

    @Override
    protected Instances determineOutputFormat(Instances inputFormat) throws Exception {
        return new Instances(inputFormat, 0);
    }

    @Override
    protected Instances process(Instances instances) throws Exception {

        // We only modify the first batch of data that this filter receives
        // (e.g., the training data of a classifier, not the test data)
        if (!isFirstBatchDone()) {

	    //
	    // INSERT YOUR CODE HERE
	    //
            HashMap<Integer, Double> firstSumResults = new HashMap<Integer, Double>();
            HashMap<Integer, Double> secondSumResults = new HashMap<Integer, Double>();
            Instance previousBest = null;

            m_Kernel.buildKernel(instances);

            int numberOfInstances = (int)((instances.size() * m_SampleSizePercent) / 100);

            Instances newInstances = new Instances(instances, 0, 0);
            for(int i = 0; i < numberOfInstances; i ++){
                Instance best = getBestInstance(instances, newInstances, firstSumResults, secondSumResults, previousBest);
                previousBest = best;
                newInstances.add(best);
            }

            return newInstances;
        }
        return instances;
    }

    // Gets the best instance given the current instances
    private Instance getBestInstance(Instances instances, Instances selectedInstances, HashMap<Integer, Double> firstSumResults
            , HashMap<Integer, Double> secondSumResults, Instance previousBest){
        Instance bestChoice = null;
        double bestValue = 0;

        for(int i = 0; i < instances.size(); i++){
            double a = getFirstTerm(i, instances, firstSumResults);
            double b = unalikeAllSelectedInstances(i, selectedInstances, secondSumResults, previousBest);

            double newValue = a -b;
            // Sets the best choice to the first element
            if(bestChoice == null){
                bestChoice = instances.instance(i);
                bestValue = newValue;
            }
            // Checks if the new value is is larger than the previous largest one
            else if(newValue > bestValue){
                bestValue = newValue;
                bestChoice = instances.instance(i);
            }
        }

        return bestChoice;
    }

    // Gets the first term, either by calculating it, or by referring to the previous value
    private double getFirstTerm(int instanceIndex, Instances instances, HashMap<Integer, Double> firstSumResults){
        // If the values has been calculated before
        if(firstSumResults.containsKey(instanceIndex)){
            return firstSumResults.get(instanceIndex);
        }
        else{
            // Calculate the value
            double result = averageOfAllInstances(instanceIndex, instances);
            firstSumResults.put(instanceIndex, result);
            return result;
        }
    }

    // Calculates the first term of the kernel herding equation
    private double averageOfAllInstances(int instanceIndex, Instances instances){
        double result = averageOfAllInstancesWorker(instanceIndex, instances);
        return  result / instances.size();
    }

    // Calculates the sum of the kernel functions for the first term
    private double averageOfAllInstancesWorker(int instanceIndex, Instances instances){
        double result = 0;
        for(int i = 0; i < instances.size(); i ++){
            try {
                result += m_Kernel.eval(instanceIndex, i, instances.instance(instanceIndex));
            }
            catch (Exception e){

            }
        }
        return result;
    }

    // Calculates the second term of the kernel herding equation
    private double unalikeAllSelectedInstances(int instanceIndex, Instances selectedInstances
            , HashMap<Integer, Double> secondSumResults, Instance previousBest){
        double oldValue = 0;

        // If the value has been calculated before
        if(secondSumResults.containsKey(instanceIndex)){
            oldValue = secondSumResults.get(instanceIndex);
        }

        try{
            double newValue = 0;
            if(selectedInstances.size() != 0){
                newValue = oldValue + m_Kernel.eval(-1, instanceIndex, previousBest);
            }
            secondSumResults.put(instanceIndex, newValue);
            return newValue / (selectedInstances.size() + 1);
        }
        catch(Exception e){
            return Double.NaN;
        }
    }



    /**
     * The main method used for running this filter from the command-line interface.
     *
     * @param options the command-line options
     */
    public static void main(String[] options) {
        runFilter(new KernelHerding(), options);
    }
}