package edu.stanford.nlp.jcoref.dcoref.sievepasses;

public class CorefDictionaryMatch2 extends DeterministicCorefSieve {
  public CorefDictionaryMatch2(){
    super();
    flags.USE_iwithini = true;
    flags.USE_DIFFERENT_LOCATION = true;
    flags.USE_NUMBER_IN_MENTION = true;
    flags.USE_DISTANCE = true;
    flags.USE_NUMBER_ANIMACY_NE_AGREE = true;
    flags.USE_COREF_DICT = true;
    flags.USE_COREF_DICT_COL2 = true;
  }
  public CorefDictionaryMatch2(String args) {
    this();
    flags.set(args);
  }
}
