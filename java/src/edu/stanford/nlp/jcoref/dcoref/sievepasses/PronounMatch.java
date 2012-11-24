package edu.stanford.nlp.jcoref.dcoref.sievepasses;

public class PronounMatch extends DeterministicCorefSieve {
  public PronounMatch() {
    super();
    flags.USE_iwithini = true;
    flags.DO_PRONOUN = true;
  }
  public PronounMatch(String args) {
    this();
    flags.set(args);
  }
}
