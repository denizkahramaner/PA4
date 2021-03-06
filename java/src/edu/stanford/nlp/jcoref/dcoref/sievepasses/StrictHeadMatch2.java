package edu.stanford.nlp.jcoref.dcoref.sievepasses;

public class StrictHeadMatch2 extends DeterministicCorefSieve {
  public StrictHeadMatch2() {
    super();
    flags.USE_iwithini = true;
    flags.USE_INCLUSION_HEADMATCH = true;
    flags.USE_WORDS_INCLUSION = true;
  }
  public StrictHeadMatch2(String args) {
    this();
    flags.set(args);
  }
}
