package edu.stanford.nlp.jcoref.dcoref.sievepasses;

public class ExactStringMatch extends DeterministicCorefSieve {
  public ExactStringMatch() {
    super();
    flags.USE_EXACTSTRINGMATCH = true;
    flags.CLUSTER_MATCH = true;
  }
  public ExactStringMatch(String args) {
    this();
    flags.set(args);
  }
}
