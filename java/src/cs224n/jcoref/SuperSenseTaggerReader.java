package edu.stanford.nlp.jcoref;

import java.util.ArrayList;
import java.util.List;

import edu.stanford.nlp.io.IOUtils;

public class SuperSenseTaggerReader {

  private static final String defaultPath = "/scr/heeyoung/corpus/coref/jcoref/jcb_dev/jcb_dev_tagged/";

  public static class SSTToken {
    String text;
    String pos;
    String lemma;
    String supersense;
    String neType;

    public SSTToken (String text, String pos, String lemma, String supersense, String neType){
      this.text = text;
      this.pos = pos;
      this.lemma = lemma;
      this.supersense = supersense;
      this.neType = neType;
    }
    public String toString() {
      return text;
    }
  }

  public static List<List<SSTToken>> readSSTOutput(String file) {
    List<List<SSTToken>> sstOutput = new ArrayList<List<SSTToken>>();
    for(String line : IOUtils.readLines(file)) {
      if(line.equals("")) continue;
      List<SSTToken> sentence = new ArrayList<SSTToken>();
      sstOutput.add(sentence);
      String[] split = line.split(" ");
      assert(split.length % 5 == 0);
      int tokenCounts = split.length / 5;
      for(int i = 0 ; i < tokenCounts; i++) {
        sentence.add(new SSTToken(split[5*i], split[5*i+1], split[5*i+2], split[5*i+3], split[5*i+4]));
      }
    }
    return sstOutput;
  }

  public static void main(String[] args) {
    readSSTOutput(defaultPath+"topic12.tags");

  }

}
