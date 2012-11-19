package edu.stanford.nlp.jcoref.dcoref.sievepasses;

public class EntityMultipleArgumentsMatch extends DeterministicCorefSieve {
  public EntityMultipleArgumentsMatch() {
    super();
    flags.USE_iwithini = true;
    flags.ENTITY_MULTIPLE_ARG_MATCH = true;
    flags.CLUSTER_MATCH = true;
  }
  public EntityMultipleArgumentsMatch(String args) {
    this();
    flags.set(args);
  }
}