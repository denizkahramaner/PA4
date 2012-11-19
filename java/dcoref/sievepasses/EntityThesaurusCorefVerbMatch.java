package edu.stanford.nlp.jcoref.dcoref.sievepasses;

public class EntityThesaurusCorefVerbMatch extends DeterministicCorefSieve {
  public EntityThesaurusCorefVerbMatch() {
    super();
    flags.USE_COREF_VERB_FOR_ENTITY = true;
    flags.CLUSTER_MATCH = true;
  }
  public EntityThesaurusCorefVerbMatch(String args) {
    this();
    flags.set(args);
  }
}
