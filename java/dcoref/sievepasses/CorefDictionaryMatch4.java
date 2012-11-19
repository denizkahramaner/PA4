package edu.stanford.nlp.jcoref.dcoref.sievepasses;

public class CorefDictionaryMatch4 extends DeterministicCorefSieve {
  public CorefDictionaryMatch4(){
    super();
    flags.USE_iwithini = true;
    flags.USE_DIFFERENT_LOCATION = true;
    flags.USE_NUMBER_IN_MENTION = true;
    flags.USE_DISTANCE = true;
    flags.USE_NUMBER_ANIMACY_NE_AGREE = true;
    flags.USE_COREF_DICT = true;
    flags.USE_COREF_DICT_COL4 = true;
  }
  public CorefDictionaryMatch4(String args) {
    this();
    flags.set(args);
  }
}
