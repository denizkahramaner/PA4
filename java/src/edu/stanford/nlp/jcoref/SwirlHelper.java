package edu.stanford.nlp.jcoref;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.jcoref.dcoref.CoNLL2011DocumentReader;
import edu.stanford.nlp.jcoref.dcoref.Constants;
import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IntPair;

/** Helper class for make input or read output for Swirl-1.1.0 (Semantic Role Labeling System) */
public class SwirlHelper {

  public static final String jcbSrlPath = "/scr/heeyoung/corpus/coref/jcoref/srl/jcb/output/";
  public static final String conllSrlPath = "/scr/heeyoung/corpus/coref/jcoref/srl/conll/conllTrainSwirlOutput/";

  public static List<Map<Integer, Map<IntPair, String>>> readSRLOutput(String file) {
    List<Map<Integer, Map<IntPair, String>>> srlInfo = new ArrayList<Map<Integer, Map<IntPair, String>>>();
    List<String[]> sentence = new ArrayList<String[]>();
    for(String line : IOUtils.readLines(file)) {
      if(line.equals("")) {
        srlInfo.add(getSentenceSrlInfo(sentence));
        sentence = new ArrayList<String[]>();
        continue;
      }
      sentence.add(line.split("\t\t"));
    }

    return srlInfo;
  }

  private static Map<Integer, Map<IntPair, String>> getSentenceSrlInfo(List<String[]> sentence) {
    Map<Integer, Map<IntPair, String>> sentSrlInfo = new HashMap<Integer, Map<IntPair, String>>();
    List<Integer> verbIdx = new ArrayList<Integer>();
    for(int col = 0 ; col < sentence.get(0).length ; col++) {
      String argTag = null;
      int argBeginIdx = -1;
      int argEndIdx = -1;
      for(int idx = 0 ; idx < sentence.size() ; idx++) {
        String str = sentence.get(idx)[col];
        if(col==0) {
          if(!str.equals("-")) {
            verbIdx.add(idx);
            sentSrlInfo.put(idx, new HashMap<IntPair, String>());
          }
        } else {
          if(str.equals("*")) continue;
          if(str.startsWith("(")) {
            argTag = (str.endsWith(")"))? str.substring(1,str.length()-2) : str.substring(1,str.length()-1);
            argBeginIdx = idx;
          }
          if(str.endsWith(")")) {
            argEndIdx = idx+1;
            sentSrlInfo.get(verbIdx.get(col-1)).put(new IntPair(argBeginIdx, argEndIdx), argTag);
          }
        }
      }
    }
    return sentSrlInfo;
  }

  public static String makeSRLInput(List<CoreMap> sentences) {
    StringBuilder sb = new StringBuilder();
    for(CoreMap sent : sentences) {
      sb.append("1");   // see README in swirl-1.1.0
      String previousNE = "O";
      for(CoreLabel token : sent.get(TokensAnnotation.class)) {
        String curNE = token.get(NamedEntityTagAnnotation.class);
        String ne = getNE(previousNE, curNE);
        previousNE = curNE;

        sb.append(" ").append(token.get(TextAnnotation.class));
        sb.append(" ").append(token.get(PartOfSpeechAnnotation.class));
        sb.append(" ").append(ne);
      }
      sb.append("\n");
    }

    return sb.toString();
  }

  private static String getNE(String previousNE, String curNE) {
    if(curNE.equals("O")) {
      return "O";
    }
    StringBuilder ne = new StringBuilder();
    if(previousNE.equals(curNE)) {
      ne.append("I-");
    } else {
      ne.append("B-");
    }
    if(curNE.startsWith("PER")) ne.append("PER");
    else if(curNE.startsWith("LOC")) ne.append("LOC");
    else if(curNE.startsWith("ORG")) ne.append("ORG");
    else if(curNE.startsWith("MISC")) ne.append("MISC");
    else {  // TODO other tags?
      return "O";
    }

    return ne.toString();
  }

  private static void jcbToSwirlInput() throws IOException{
    String jcbPath = "/scr/heeyoung/corpus/coref/jcoref/jcb_test/";
    String outputPath = "/scr/heeyoung/corpus/coref/jcoref/srl/jcb/test_input/";
    JCBReader jcbReader = new JCBReader(jcbPath);
    Properties props = new Properties();
    props.put("annotators", "pos, lemma, ner");
    props.put("ner.useSUTime", "false");
    StanfordCoreNLP stanfordProcessor = new StanfordCoreNLP(props, false);

    JCBDocument doc;
    while((doc=jcbReader.nextDoc())!=null){
      stanfordProcessor.annotate(doc.annotation);
      String str = makeSRLInput(doc.annotation.get(SentencesAnnotation.class));
      PrintWriter pw = IOUtils.getPrintWriter(outputPath+"swirlInput"+doc.docID+".txt");
      pw.print(str);
      pw.close();
    }
  }
  private static void conllToSwirlInput() throws Exception {
    // read conll corpus
    String corpusPath = "/scr/nlp/data/conll-2011/v2/data/dev/data/english/annotations";
    String outputPath = "/scr/heeyoung/corpus/coref/jcoref/srl/conllSwirlInput/";

    CoNLL2011DocumentReader.Options options = new CoNLL2011DocumentReader.Options();
    options.annotateTokenCoref = false;
    options.annotateTokenSpeaker = Constants.USE_GOLD_SPEAKER_TAGS;
    options.annotateTokenNer = Constants.USE_GOLD_NE;
    options.annotateTokenPos = Constants.USE_GOLD_POS;
    if (Constants.USE_CONLL_AUTO) options.setFilter(".*_auto_conll$");

    CoNLL2011DocumentReader reader = new CoNLL2011DocumentReader(corpusPath, options);
    CoNLL2011DocumentReader.Document conllDoc;

    while((conllDoc = reader.getNextDocument())!=null) {
      String str = makeSRLInput(conllDoc.getAnnotation().get(SentencesAnnotation.class));
      String docID = conllDoc.getDocumentID().replaceAll("/", "-")+"-part-"+conllDoc.getPartNo();
      PrintWriter pw = IOUtils.getPrintWriter(outputPath+"swirlInput-"+docID+".txt");
      pw.print(str);
      pw.close();
    }
  }

  public static void main(String[] args) throws Exception {
    //    String file = "/home/heeyoung/corpus-jcb/srl/newOutput/swirlOutput6-8.txt";
    //    List<Map<Integer, Map<IntPair, String>>> srls = readSRLOutput(file);
    //    System.err.println();

    jcbToSwirlInput();
    //    conllToSwirlInput();
  }
}
