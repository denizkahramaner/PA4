package edu.stanford.nlp.jcoref;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Stack;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.jcoref.dcoref.Mention;
import edu.stanford.nlp.ling.CoreAnnotations.IndexAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.OriginalTextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.UtteranceAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.objectbank.TokenizerFactory;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.AbstractIterator;
import edu.stanford.nlp.util.CoreMap;

public class JCBReader implements Serializable {

  protected List<File> fileList;
  private int curFileIndex;
  private final boolean sortFiles = false;
  private DocumentIterator docIterator = null;
  private final TokenizerFactory<CoreLabel> tokenizerFactory;
  protected int maxGoldMentionID = 0;
  private int maxGoldClusterID = 0;
  private static final boolean readAllDocsInTopic = false;  // changed back to false

  // cluster ID conversion Map
  private final Map<String, Integer> clusterIDStr2Int = new HashMap<String, Integer>();
  private final static Map<Integer, String> clusterIDInt2Str = new HashMap<Integer, String>();
  protected Counter<Integer> mentionCounterInCluster = new ClassicCounter<Integer>();

  public static Logger logger = Logger.getLogger(JCBReader.class.getName());

  public JCBReader(String filepath) {
    this(filepath, ".jcb");
  }
  public JCBReader(String filepath, String filter) {
    this.fileList = getFiles(filepath, filter);
    if(sortFiles) {
      Collections.sort(this.fileList);
    }
    curFileIndex = 0;
    tokenizerFactory = PTBTokenizer.factory(false, new CoreLabelTokenFactory(false));
  }

  private static List<File> getFiles(String filepath, String ext) {
    Iterable<File> iter = IOUtils.iterFilesRecursive(new File(filepath), ext);
    List<File> fileList = new ArrayList<File>();
    for (File f:iter) {
      fileList.add(f);
    }
    return fileList;
  }

  public JCBDocument nextDoc(){
    JCBDocument doc = null;

    try {
      while(curFileIndex < fileList.size()) {
        File curFile = fileList.get(curFileIndex);
        if (docIterator == null) {
          docIterator = new DocumentIterator(curFile.getAbsolutePath(), tokenizerFactory);
          RuleBasedJointCorefSystem.logger.fine("Read doc: "+curFile.toString());
        }
        doc = docIterator.next();
        //        int dothis = 5;
        //        RuleBasedJointCorefSystem.logger.fine("do this: "+dothis);
        //        if(doc!=null && doc.topicID % 6 != dothis) doc = null;

        if(doc==null) {
          curFileIndex++;
          docIterator = null;
        }
        else return doc;
      }
      return null;
    } catch (IOException ex) {
      throw new RuntimeException(ex);
    }
  }
  private class DocumentIterator extends AbstractIterator<JCBDocument> {
    int currentDoc;
    int docCnt = 0;
    List<JCBDocument> docs;

    public DocumentIterator(String filename, TokenizerFactory<CoreLabel> tokenizerFactory) throws IOException {
      currentDoc = 0;
      docs = readDocuments(filename, tokenizerFactory);
    }

    @Override
    public boolean hasNext() {
      return (currentDoc < docCnt);
    }

    @Override
    public JCBDocument next() {
      if(hasNext()) return docs.get(currentDoc++);
      else return null;
    }

    // read all docs in the file and return list of JointCorefDocument
    public List<JCBDocument> readDocuments(String filename, TokenizerFactory<CoreLabel> tokenizerFactory) throws IOException{
      List<JCBDocument> docs = new ArrayList<JCBDocument>();
      JCBDocument doc = null;
      int topicID = -1;
      int docID = -1;
      List<CoreMap> sentences = null;
      StringBuilder docText = null;

      for(String line : IOUtils.readLines(filename)){

        if(line.startsWith("<TOPIC") || line.startsWith("</TOPIC")
            || line.startsWith("<DOC") || line.startsWith("</DOC")) {
          if(readAllDocsInTopic) {
            if(line.startsWith("<TOPIC")) {
              Pattern topicIDPattern = Pattern.compile("<TOPIC (.*)>", Pattern.DOTALL+Pattern.CASE_INSENSITIVE);
              Matcher topicIDMatcher = topicIDPattern.matcher(line);
              if(topicIDMatcher.find()) topicID = Integer.parseInt(topicIDMatcher.group(1));
              doc = new JCBDocument(topicID+"-"+docID);
              docs.add(doc);
              docCnt++;
              docText = new StringBuilder();
              sentences = doc.annotation.get(SentencesAnnotation.class);
            } else if(line.startsWith("</TOPIC")) {
              doc.annotation.set(TextAnnotation.class, docText.toString().trim());
            } else {

              // do nothing

              //              // add blank line for <DOC> or </DOC>
              //              docText.append("\n");
              //              CoreMap sent = new Annotation("\n");
              //              List<CoreLabel> sentence = new ArrayList<CoreLabel>();
              //              sentence.add(new CoreLabel());
              //              sent.set(TokensAnnotation.class, sentence);
              //              sentences.add(sent);
              //              doc.goldMentions.add(new ArrayList<Mention>());
            }
          } else {
            if(line.startsWith("<TOPIC")) {
              Pattern topicIDPattern = Pattern.compile("<TOPIC (.*)>", Pattern.DOTALL+Pattern.CASE_INSENSITIVE);
              Matcher topicIDMatcher = topicIDPattern.matcher(line);
              if(topicIDMatcher.find()) topicID = Integer.parseInt(topicIDMatcher.group(1));
            } else if(line.startsWith("</TOPIC")) {
              // do nothing
            } else if(line.startsWith("</DOC")) {
              doc.annotation.set(TextAnnotation.class, docText.toString().trim());
            } else if(line.startsWith("<DOC")) {
              Pattern docIDPattern = Pattern.compile("<DOC (.*?).ecb>");
              Matcher docIDMatcher = docIDPattern.matcher(line);
              if(docIDMatcher.find()) docID = Integer.parseInt(docIDMatcher.group(1));
              doc = new JCBDocument(topicID+"-"+ docID);
              docs.add(doc);
              docCnt++;
              docText = new StringBuilder();
              sentences = doc.annotation.get(SentencesAnnotation.class);
            }
          }
        } else {  // sentence

          List<CoreLabel> words = tokenizerFactory.getTokenizer(new StringReader(line)).tokenize();
          List<CoreLabel> sentence = new ArrayList<CoreLabel>();
          Stack<Mention> entityStack = new Stack<Mention>();
          Stack<Mention> eventStack = new Stack<Mention>();
          List<Mention> mentions = new ArrayList<Mention>();
          StringBuilder sentText = new StringBuilder();

          // mention history for discontinuous mentions
          Map<Integer, Mention> recentMentions = new HashMap<Integer, Mention>();

          Pattern entityIDPattern = Pattern.compile("COREFID=\"(.*?)\">");
          Pattern eventIDPattern = Pattern.compile("CHAIN=\"(.*?)\">");

          for (CoreLabel word : words) {
            String w = word.get(TextAnnotation.class);
            if (!w.startsWith("<")) {
              word.remove(OriginalTextAnnotation.class);
              sentence.add(word);
              word.set(IndexAnnotation.class, sentence.size());
              word.set(UtteranceAnnotation.class, 0);
              sentText.append(w).append(" ");
            } else if (w.startsWith("<Entity")) {
              Mention mention = new Mention();
              mention.sentNum = sentences.size();
              mention.startIndex = sentence.size();
              Matcher entityIDMatcher = entityIDPattern.matcher(w);
              if(entityIDMatcher.find()) {
                String strClusterID = "T"+topicID+"-C"+Integer.parseInt(entityIDMatcher.group(1));
                if(clusterIDStr2Int.containsKey(strClusterID)) {
                  mention.goldCorefClusterID = clusterIDStr2Int.get(strClusterID);
                } else {
                  mention.goldCorefClusterID = ++maxGoldClusterID;
                  clusterIDStr2Int.put(strClusterID, mention.goldCorefClusterID);
                  clusterIDInt2Str.put(mention.goldCorefClusterID, strClusterID);
                }
                mentionCounterInCluster.incrementCount(mention.goldCorefClusterID);
              }
              mention.mentionID = ++maxGoldMentionID;
              if(doc.maxIDGoldMention < mention.mentionID) doc.maxIDGoldMention = mention.mentionID;
              entityStack.push(mention);
            } else if (w.startsWith("<MENTION")) {
              Matcher eventIDMatcher = eventIDPattern.matcher(w);
              if(eventIDMatcher.find()) {
                String matchedID = eventIDMatcher.group(1);
                if(matchedID.endsWith("*")) {
                  int id = Integer.parseInt(matchedID.substring(0, matchedID.length()-1));
                  String strClusterID = "T"+topicID+"-C"+id;
                  Mention mention = recentMentions.get(clusterIDStr2Int.get(strClusterID));
                  mention.additionalSpanStartIndex = sentence.size();
                  eventStack.push(mention);
                } else {
                  Mention mention = new Mention();
                  mention.isEvent = true;
                  mention.startIndex = sentence.size();
                  String strClusterID = "T"+topicID+"-C"+Integer.parseInt(matchedID);
                  if(clusterIDStr2Int.containsKey(strClusterID)) {
                    mention.goldCorefClusterID = clusterIDStr2Int.get(strClusterID);
                  } else {
                    mention.goldCorefClusterID = ++maxGoldClusterID;
                    clusterIDStr2Int.put(strClusterID, mention.goldCorefClusterID);
                    clusterIDInt2Str.put(mention.goldCorefClusterID, strClusterID);
                  }
                  mentionCounterInCluster.incrementCount(mention.goldCorefClusterID);
                  mention.mentionID = ++maxGoldMentionID;
                  if(doc.maxIDGoldMention < mention.mentionID) doc.maxIDGoldMention = mention.mentionID;
                  recentMentions.put(mention.goldCorefClusterID, mention);
                  eventStack.push(mention);
                }
              }
            } else if (w.startsWith("</Entity")) {
              Mention mention = entityStack.pop();
              mention.endIndex = sentence.size();
              mentions.add(mention);
            } else if (w.startsWith("</MENTION")) {
              Mention mention = eventStack.pop();
              if(mention.additionalSpanStartIndex == -1) {
                mention.endIndex = sentence.size();
                mentions.add(mention);
              } else {
                mention.additionalSpanEndIndex = sentence.size();
              }
            }
          }
          docText.append(sentText);
          CoreMap sent = new Annotation(sentText.toString().trim());
          sent.set(TokensAnnotation.class, sentence);
          sentences.add(sent);
          doc.goldMentions.add(mentions);
        }
      }
      return docs;
    }
  }

  public Map<String, JCBDocument> readInputFiles() {
    Map<String, JCBDocument> inputs = new HashMap<String, JCBDocument>();

    JCBDocument doc;
    while((doc = this.nextDoc())!=null) {
      // read SRL info
      doc.srlInfo = SwirlHelper.readSRLOutput(SwirlHelper.jcbSrlPath+"swirlOutput"+doc.docID+".txt");
      inputs.put(doc.docID, doc);
    }
    return inputs;
  }


  public static void main(String[] args) {
    String jcbPath = "/scr/heeyoung/corpus/coref/jcoref/jcb_v0.3/";
    JCBReader jr = new JCBReader(jcbPath);
    Map<String, JCBDocument> docs = jr.readInputFiles();

    StringBuilder sb = new StringBuilder();
    sb.append("## Entity(N) or Event(V)?");
    sb.append("\t").append("Topic");
    sb.append("\t").append("Doc");
    sb.append("\t").append("Sentence Number");
    sb.append("\t").append("CorefID");
    sb.append("\t").append("StartIdx");
    sb.append("\t").append("EndIdx");
    sb.append("\t").append("StartCharIdx");
    sb.append("\t").append("EndCharIdx");
    
    System.out.println(sb.toString());
    for(String docID : docs.keySet()) {
//      System.out.println(docID);
      JCBDocument doc = docs.get(docID);
      String topic = docID.split("-")[0];
      String document = docID.split("-")[1];
      int sentNum = 0;
      for(List<Mention> mentions : doc.goldMentions){
        List<CoreLabel> sentence = doc.annotation.get(SentencesAnnotation.class).get(sentNum).get(TokensAnnotation.class);
        for(Mention m : mentions) {
          int startCharIdx = 0;
          int endCharIdx = 0;          
          for(CoreLabel c : sentence){
            int wordLen = c.word().length();
            if(c.word().equals("-LRB-") || c.word().equals("-RRB-")) wordLen = 1;
            if(c.word().contains("\\/")) {
              wordLen--;
            }
            if((c.word().equalsIgnoreCase("neighborhood") && !topic.equals("26"))
                || c.word().equalsIgnoreCase("misdemeanor")) wordLen++;
            if(c.word().contains(" ")) wordLen--;
            // TODO remove space in wordlength
            if(c.index()-1 < m.startIndex) startCharIdx += wordLen;
            if(c.index() <= m.endIndex) endCharIdx += wordLen;
            else break;
          }
          String type = (m.isEvent)? "V" : "N";
          System.out.println(type+"\t"+topic+"\t"+document+"\t"+sentNum+"\t"+clusterIDInt2Str.get(m.goldCorefClusterID).split("-C")[1]+"\t"+m.startIndex+"\t"+m.endIndex+"\t"+startCharIdx+"\t"+endCharIdx);
          if(m.additionalSpanStartIndex!=-1) {
            startCharIdx = 0;
            endCharIdx = 0;
            for(CoreLabel c : sentence){
              if(c.index()-1 < m.additionalSpanStartIndex) startCharIdx += c.word().length();
              if(c.index() <= m.additionalSpanEndIndex) endCharIdx += c.word().length();
              else break;
            }
            System.out.println(type+"\t"+topic+"\t"+document+"\t"+sentNum+"\t"+clusterIDInt2Str.get(m.goldCorefClusterID).split("-C")[1]+"*\t"+m.additionalSpanStartIndex+"\t"+m.additionalSpanEndIndex+"\t"+startCharIdx+"\t"+endCharIdx);
          }
        }
        sentNum++;
      }
    }
//    System.err.println();
  }
}
