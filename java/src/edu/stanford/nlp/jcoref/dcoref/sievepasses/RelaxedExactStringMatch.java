package edu.stanford.nlp.jcoref.dcoref.sievepasses;

public class RelaxedExactStringMatch extends DeterministicCorefSieve {
  public RelaxedExactStringMatch() {
    super();
    flags.USE_RELAXED_EXACTSTRINGMATCH = true;
    flags.USE_NUMBER_IN_MENTION = true;
  }
  public RelaxedExactStringMatch(String args) {
    this();
    flags.set(args);
  }
}
