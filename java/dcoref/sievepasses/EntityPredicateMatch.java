package edu.stanford.nlp.jcoref.dcoref.sievepasses;

public class EntityPredicateMatch extends DeterministicCorefSieve {
  public EntityPredicateMatch() {
    super();
    flags.ENTITY_PREDICATE_MATCH = true;
    flags.CLUSTER_MATCH = true;
  }
  public EntityPredicateMatch(String args) {
    this();
    flags.set(args);
  }
}