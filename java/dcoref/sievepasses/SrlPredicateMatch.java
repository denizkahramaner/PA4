package edu.stanford.nlp.jcoref.dcoref.sievepasses;

public class SrlPredicateMatch extends DeterministicCorefSieve {
  public SrlPredicateMatch() {
    super();
    flags.USE_COREF_SRLPREDICATE_MATCH = true;
    flags.CLUSTER_MATCH = true;
    flags.USE_ATTRIBUTES_AGREE = true;
  }
  public SrlPredicateMatch(String args) {
    this();
    flags.set(args);
  }
}
