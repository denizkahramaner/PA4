package edu.stanford.nlp.jcoref;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.jcoref.dcoref.Mention;
import edu.stanford.nlp.ling.CoreAnnotations.IndexAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.util.CoreMap;

public class BejanCoNLLOutput {
  private static final String bejanOutPath = "/scr/heeyoung/coref/event/hdpflat4mihai/out/";
  private static final String bejanData = "/scr/heeyoung/coref/event/hdpflat4mihai/in/st_01_proc2cluster_v011/cvman/cv_man_fold1.test";
  private static final String headAlphabet = "/scr/heeyoung/coref/event/hdpflat4mihai/in/st_01_proc2cluster_v011/alphabets/head.logalph";
  private static final String jcbPath = "/scr/heeyoung/corpus/coref/jcoref/jcb_v0.1/";
  private static final String goldOutput = "/scr/heeyoung/coref/event/hdpflat4mihai/conlloutput/goldOutput.txt";
  private static final String bejanOutput = "/scr/heeyoung/coref/event/hdpflat4mihai/conlloutput/bejanOutput.txt";
  private final Map<Integer, String> headMap = new HashMap<Integer, String>();
  private final Map<String, Integer> mentionToClusterID = new HashMap<String, Integer>();
  private final Map<String, String> positionToMention = new HashMap<String, String>();
  private final Map<String, JCBDocument> topicDocIDToJCBDocument = new HashMap<String, JCBDocument>();
  private final Set<String> topicDocIDsInBejanOutput = new HashSet<String>();

  public static void main(String[] args) throws IOException {
    BejanCoNLLOutput bco = new BejanCoNLLOutput();
    bco.readJCBDocuments(jcbPath);
    bco.readHeadMap(headAlphabet);
    bco.readBejanData(bejanData);
    bco.readAllBejanOutput(bejanOutPath);
    bco.printCoNLLOutput();
  }

  public void readAllBejanOutput(String path) {
    for(File file : IOUtils.iterFilesRecursive(new File(bejanOutPath), "sysibp")){
      readBejanOutput(file);
    }
  }

  public void readBejanOutput(File file) {
    String[] tmp = file.toString().split("MAN_")[1].split("\\.")[0].split("_");
    int topicID = Integer.parseInt(tmp[0]);
    int docID = Integer.parseInt(tmp[1]);
    topicDocIDsInBejanOutput.add(topicID+"_"+docID);

    for(String line : IOUtils.readLines(file)) {
      String[] split = line.split(" ");
      int clusterID = Integer.parseInt(split[0]);
      for(int i = 1 ; i < split.length ; i++){
        mentionToClusterID.put(topicID+"_"+docID+"_"+split[i], clusterID);
      }
    }
  }

  public void readHeadMap(String file) {
    for(String line : IOUtils.readLines(file)) {
      String[] split = line.split("\\s+");
      headMap.put(Integer.parseInt(split[1]), split[0]);
    }
  }

  public void readJCBDocuments(String path) {
    JCBReader jcbReader = new JCBReader(jcbPath);
    JCBDocument jDoc = null;
    while((jDoc = jcbReader.nextDoc())!=null) {
      topicDocIDToJCBDocument.put(jDoc.docID, jDoc);
    }
  }

  public void readBejanData(String file) {
    String p = "<e><eid>([^<]*)</eid><did>MAN_([^<]*)</did><sid>([^<]*)</sid><bnd>([^<]*)</bnd><head>([^<]*)</head>";
    Pattern pattern = Pattern.compile(p);
    //    int lineCount = 0;
    for(String line : IOUtils.readLines(file)) {
      //      System.err.println(lineCount++);
      Matcher matcher = pattern.matcher(line);
      while(matcher.find()) {
        String menID = matcher.group(1);
        String topicDocID = matcher.group(2);
        int sentNum = Integer.parseInt(matcher.group(3).split("_")[1])-1;
        int headIdx = Integer.parseInt(matcher.group(4).split(":")[0]);
        String headText = headMap.get(Integer.parseInt(matcher.group(5)));

        positionToMention.put(topicDocID+"_"+sentNum+"_"+headIdx, topicDocID+"::"+menID+"::"+headText);
      }
    }
  }

  public void printCoNLLOutput() throws FileNotFoundException {
    PrintWriter goldOutputWriter = new PrintWriter(new FileOutputStream(goldOutput));
    PrintWriter bejanOutputWriter = new PrintWriter(new FileOutputStream(bejanOutput));

    int countMismatch = 0;
    int countDoc = 0;
    for(int t=1 ; t < 46 ; t++) {

      StringBuilder goldSb = new StringBuilder();
      StringBuilder bejanSb = new StringBuilder();

      goldSb.append("#begin document ").append(t).append("\n");
      bejanSb.append("#begin document ").append(t).append("\n");

      for(String topicDocID : topicDocIDToJCBDocument.keySet()) {
        if(!topicDocIDsInBejanOutput.contains(topicDocID)) {
          //        System.err.println("not exist!! "+topicDocID);
          continue;
        }
        int topic = Integer.parseInt(topicDocID.split("_")[0]);
        if(t!=topic) continue;



        System.err.println(countDoc++ +" doc processed");

        JCBDocument jdoc = topicDocIDToJCBDocument.get(topicDocID);
        Annotation anno = jdoc.annotation;
        List<CoreMap> sentences = anno.get(SentencesAnnotation.class);
        for(int sentNum = 0; sentNum < sentences.size() ; sentNum++) {
          List<CoreLabel> sentence = sentences.get(sentNum).get(TokensAnnotation.class);
          List<Mention> goldMentions = jdoc.goldMentions.get(sentNum);
          Map<Integer, Integer> eventMentionIndex = new HashMap<Integer, Integer>();
          for(Mention gold : goldMentions) {
            if(gold.isEvent) {
              eventMentionIndex.put(gold.startIndex, gold.goldCorefClusterID);
            }
          }
          for(CoreLabel word : sentence) {
            int wordIdx = word.get(IndexAnnotation.class);

            String positionCorrect = jdoc.docID+"_"+sentNum+"_"+(wordIdx-1);
            String positionNext = jdoc.docID+"_"+sentNum+"_"+(wordIdx);
            String position2ndNext = jdoc.docID+"_"+sentNum+"_"+(wordIdx+1);
            String positionPrevious = jdoc.docID+"_"+sentNum+"_"+(wordIdx-2);
            String position2ndPrevious = jdoc.docID+"_"+sentNum+"_"+(wordIdx-3);

            goldSb.append(wordIdx).append("\t").append(word.get(TextAnnotation.class)).append("\t");
            bejanSb.append(wordIdx).append("\t").append(word.get(TextAnnotation.class)).append("\t");
            if(eventMentionIndex.containsKey(wordIdx-1)) {    // gold mention exist
              goldSb.append("(").append(eventMentionIndex.get(wordIdx-1)).append(")\n");

              String eventWord = word.get(TextAnnotation.class);
              if(positionToMention.containsKey(positionCorrect) && eventWord.equals(positionToMention.get(positionCorrect).split("::")[2])) {
                String[] mentionInfo = positionToMention.get(positionCorrect).split("::");
                bejanSb.append("(").append(mentionToClusterID.get(mentionInfo[0]+"_"+mentionInfo[1])).append(")\n");
                continue;
              } else if(positionToMention.containsKey(positionNext) && eventWord.equals(positionToMention.get(positionNext).split("::")[2])) {
                String[] mentionInfo = positionToMention.get(positionNext).split("::");
                bejanSb.append("(").append(mentionToClusterID.get(mentionInfo[0]+"_"+mentionInfo[1])).append(")\n");
                continue;
              } else if(positionToMention.containsKey(position2ndNext) && eventWord.equals(positionToMention.get(position2ndNext).split("::")[2])) {
                String[] mentionInfo = positionToMention.get(position2ndNext).split("::");
                bejanSb.append("(").append(mentionToClusterID.get(mentionInfo[0]+"_"+mentionInfo[1])).append(")\n");
                continue;
              } else if(positionToMention.containsKey(positionPrevious) && eventWord.equals(positionToMention.get(positionPrevious).split("::")[2])) {
                String[] mentionInfo = positionToMention.get(positionPrevious).split("::");
                bejanSb.append("(").append(mentionToClusterID.get(mentionInfo[0]+"_"+mentionInfo[1])).append(")\n");
                continue;
              } else if(positionToMention.containsKey(position2ndPrevious) && eventWord.equals(positionToMention.get(position2ndPrevious).split("::")[2])) {
                String[] mentionInfo = positionToMention.get(position2ndPrevious).split("::");
                bejanSb.append("(").append(mentionToClusterID.get(mentionInfo[0]+"_"+mentionInfo[1])).append(")\n");
                continue;
              } else {
                // TODO : 49 missed
                System.err.println("cannot find:"+countMismatch++);
                bejanSb.append("-\n");
              }
            } else {
              goldSb.append("-\n");
              bejanSb.append("-\n");
            }
          }
          goldSb.append("\n");
          bejanSb.append("\n");
        }

      }

      goldSb.append("#end document ").append(t).append("\n");
      bejanSb.append("#end document ").append(t).append("\n");

      goldOutputWriter.print(goldSb.toString());
      bejanOutputWriter.print(bejanSb.toString());

    }

    goldOutputWriter.close();
    bejanOutputWriter.close();

    System.err.println("");
  }

}
