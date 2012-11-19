package edu.stanford.nlp.jcoref.dcoref.sievepasses;

public class EntityThesaurusMatch extends DeterministicCorefSieve {
  public EntityThesaurusMatch() {
    super();
    flags.USE_THESAURUS_ENTITY_SIMILAR = true;
    flags.CLUSTER_MATCH = true;
  }
  public EntityThesaurusMatch(String args) {
    this();
    flags.set(args);
  }
}
