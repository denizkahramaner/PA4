package edu.stanford.nlp.jcoref.dcoref.sievepasses;

public class RelaxedHeadMatch extends DeterministicCorefSieve {
  public RelaxedHeadMatch() {
    super();
    flags.USE_iwithini = true;
    flags.USE_RELAXED_HEADMATCH = true;
    flags.USE_WORDS_INCLUSION = true;
    flags.USE_ATTRIBUTES_AGREE = true;
  }
  public RelaxedHeadMatch(String args) {
    this();
    flags.set(args);
  }
}