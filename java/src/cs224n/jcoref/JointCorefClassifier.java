package edu.stanford.nlp.jcoref;

import java.io.IOException;
import java.io.Serializable;

import edu.stanford.nlp.classify.GeneralDataset;
import edu.stanford.nlp.classify.LinearRegressionFactory;
import edu.stanford.nlp.classify.LinearRegressor;
import edu.stanford.nlp.classify.RVFDataset;
import edu.stanford.nlp.classify.Regressor;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.ling.RVFDatum;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.Index;

public class JointCorefClassifier implements Serializable {

  private static final long serialVersionUID = 4747757730342985167L;

  public Regressor<String> regressor = null;

  public GeneralDataset<Double, String> trainData = null;

  public JointCorefClassifier() {
    trainData = new RVFDataset<Double, String>();
  }

  public void addData(RVFDatum<Double, String> datum) {
    trainData.add(datum);
  }

  public void train(double oldCoefficientWeight) {

    LinearRegressionFactory<String> linearRegressionFactory = new LinearRegressionFactory<String>();

    if(regressor == null) {
      regressor = linearRegressionFactory.train(trainData);
    } else {
      Regressor<String> newRegressor = linearRegressionFactory.train(trainData);
      
      // average
      Index<String> oldFeatureIndex = ((LinearRegressor<String>)regressor).featureIndex();
      Index<String> newFeatureIndex = ((LinearRegressor<String>)newRegressor).featureIndex();
      
      double[] oldWeights = ((LinearRegressor<String>)regressor).weights();
      double[] newWeights = ((LinearRegressor<String>)newRegressor).weights();

      for(int index = 0 ; index < Math.min(oldFeatureIndex.size(), newFeatureIndex.size()) ; index++) {
        String feature = oldFeatureIndex.get(index);
        int newIndex = newFeatureIndex.indexOf(feature);
        if(newIndex >= 0) oldWeights[index] = oldWeights[index]*oldCoefficientWeight + newWeights[newIndex]*(1-oldCoefficientWeight);
        else oldWeights[index] = oldWeights[index]*oldCoefficientWeight;
      }
      regressor = new LinearRegressor<String>(oldWeights, oldFeatureIndex);
    }
    
    
    RuleBasedJointCorefSystem.logger.fine("data size: "+trainData.size());
    Counter<Double> counts = new ClassicCounter<Double>();
    for(Datum<Double, String> datum : trainData) {
      counts.incrementCount(Math.floor(datum.label()*10));
    }
    RuleBasedJointCorefSystem.logger.fine(counts.toString());
    RuleBasedJointCorefSystem.logger.fine("Regressor features: ");
    RuleBasedJointCorefSystem.logger.fine(((LinearRegressor<String>)regressor).getTopFeatures(100).toString());
    System.err.println();
  }

  public double valueOf(Datum<Double, String> datum) {
    return regressor.valueOf(datum);
  }

  public void serialize(String filename) throws IOException {
    trainData = null;
    IOUtils.writeObjectToFile(this, filename);
  }

  public static JointCorefClassifier load(String filename) throws IOException, ClassNotFoundException {
    return IOUtils.readObjectFromFile(filename);
  }
}
