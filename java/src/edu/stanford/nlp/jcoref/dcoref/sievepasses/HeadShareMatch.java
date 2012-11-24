package edu.stanford.nlp.jcoref.dcoref.sievepasses;

public class HeadShareMatch extends DeterministicCorefSieve {
  public HeadShareMatch() {
    super();
    flags.DO_HEADSHARING = true;
  }
  public HeadShareMatch(String args) {
    this();
    flags.set(args);
  }
}
