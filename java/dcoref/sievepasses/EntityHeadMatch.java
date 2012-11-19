package edu.stanford.nlp.jcoref.dcoref.sievepasses;

public class EntityHeadMatch extends DeterministicCorefSieve {
  public EntityHeadMatch() {
    super();
    flags.USE_iwithini = true;
    flags.USE_INCLUSION_HEADMATCH = true;
  }
  public EntityHeadMatch(String args) {
    this();
    flags.set(args);
  }
}
