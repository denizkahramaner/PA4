package cs224n.coref;

import java.util.*;

import cs224n.ling.Tree;

public class HobbsCandidatesRule implements Rule{
	
	
	
	
	private Map<Tree<String>, Tree<String>> getChildParentMap(Tree<String> tree) {
		return getChildParentMapHelper(tree, new HashMap<Tree<String>, Tree<String>>());
	}
	private Map<Tree<String>, Tree<String>> getChildParentMapHelper(Tree<String> tree, Map<Tree<String>, Tree<String>> childParentMap) {
		if (tree == null)
			return childParentMap;
		
		for (Tree<String> child : tree.getChildren()) {
			childParentMap.put(child, tree);
			childParentMap = getChildParentMapHelper(child, childParentMap);
			// I don't think assigning is necessary since the same map is being mutated the whole time, but w/e
		}
		
		return childParentMap;
	}
	private boolean hasSAmongParents(Tree<String> xTree, Map<Tree<String>, Tree<String>> childToParent) {
		// returns false when
		// xTree was the highest S in the sentence. Well, more like there were no more S's in the sentence
		boolean found = false;
		Tree<String> temp = xTree;
		while(childToParent.get(temp) != null) {
			temp = childToParent.get(temp);
			if (temp.getLabel() == "S") {
				found = true;
				break;
			}
		}
		
		return found;
	}
	
	private List<Tree<String>> findNPAboveNPorS(Tree<String> head) {
		return findNPAboveNPorSHelper(head, new ArrayList<Tree<String>>());
	}
	private List<Tree<String>> findNPAboveNPorSHelper(Tree<String> head, List<Tree<String>> list) {
		if (head == null)
			return list;
		
		// check head
		if (head.getLabel().equals("NP")) {
			list.add(head);
			return list;
		} else if (head.getLabel().equals("S")) {
			return list;
		}
		
		// check children
		for (Tree<String> child : head.getChildren()) {
			list = findNPAboveNPorSHelper(child, list);
		}
		
		return list;
	}
	
	private Map<Mention, HashSet<Mention>> hobbsAlgorithm(Document doc) {
		// Let's try to do Hobb's. Every mention needs to take a look at who it could be paired with.
	    Map<Mention, HashSet<Mention>> candidateMentions = new HashMap<Mention,HashSet<Mention>>();
	    
	    Map<Tree<String>, Mention> treeMentionMap = new HashMap<Tree<String>, Mention>();
	    for (Mention m : doc.getMentions()) {
	    	treeMentionMap.put(m.parse, m);
	    }
	    
		for(Mention m : doc.getMentions()) {
			//List<Mention> hobbs = new ArrayList<Mention>();
			HashSet<Mention> hobbs = new HashSet<Mention>();
			Tree<String> sentenceTree = m.sentence.parse;
			
			Map<Tree<String>, Tree<String>> childToParent = getChildParentMap(sentenceTree);
			
			// 1 Begin at NP of our mention
			Tree<String> mentionTree = m.parse;
			
			// 2 Go up the tree to the first S or NP, this is called the xTree
			//   The path used pPath will just be the node whose parent is xTree
			Tree<String> xTree = childToParent.get(mentionTree);
			Tree<String> pPath = mentionTree;
			while (xTree != null && !xTree.getLabel().equals("NP") && !xTree.getLabel().equals("S")) {
				pPath = xTree;
				xTree = childToParent.get(xTree);
			}
			if (xTree == null) {
				//System.out.println("2: hit top of sentence w/o finding another NP or S");
				continue; // Well there's nothing we can do we hit the top of the sentence already...
			}
			
			// 3 Search the tree left of pPath breadth first
			for (Tree<String> child : xTree.getChildren()) {
				// We have gone as far right as possible if...
				if (child == pPath)
					break;
				
				// Search to your heart's content for any NP. These work if you can find a parent that isn't X that is NP or S
				for (Tree<String> node : child.getPreOrderTraversal()) {
					if (node.getLabel().equals("NP")) {
						// Great. Now does it have a parent that is either NP or S (before hitting X?)
						
						boolean found = false;
						Tree<String> parent = node;
						while (!parent.equals(xTree) && !found) {
							parent = childToParent.get(parent);
							
							if (parent == null) // NOTE: this probably should not be needed...
								break;
							
							if (parent.getLabel().equals("NP") || parent.getLabel().equals("S")) {
								found = true;
							}
						}
						
						if (found) {
							// we found a candidate! Get the mention for the tree and add it!
							Mention m2 = treeMentionMap.get(node);
							if (m2 != null) { hobbs.add(m2); }
						}
					}
				}
			}
			
			// 4 decide if you need to enter this loop
			while (hasSAmongParents(xTree, childToParent)){
				// 5-9 while loop
				
				// 5 Go up the tree to the first S or NP, this is called the xTree
				//   The path used pPath will just be the node whose parent is xTree
				xTree = childToParent.get(xTree);
				pPath = mentionTree;
				while (xTree != null && !xTree.getLabel().equals("NP") && !xTree.getLabel().equals("S")) {
					pPath = xTree;
					xTree = childToParent.get(xTree);
				}
				if (xTree == null) {
					//System.out.println("5: hit top of sentence w/o finding another NP or S. BAD");
					continue; // Well there's nothing we can do we hit the top of the sentence already...
					// NOTE THAT THIS ONE SHOULDN"T HAPPEN
				}
				
				// 6 If X and the path passes through the nominal that dominates the mention, don't mark it. Else, mark it.
				//   It seems like it'll never do this, so let's mark it anyway for now
				if (xTree.getLabel().equals("NP")) { // requires NP
					Mention m2 = treeMentionMap.get(xTree);
					if (m2 != null) { hobbs.add(m2); }
				}
				
				// 7 Traverse branches to the left of p, breadth first. Any NP counts as an antecedent
				for (Tree<String> child : xTree.getChildren()) {
					// We have gone as far right as possible if...
					if (child == pPath)
						break;
					
					// Search to your heart's content for any NP. These work if you can find a parent that isn't X that is NP or S
					for (Tree<String> node : child.getPreOrderTraversal()) {
						if (node.getLabel().equals("NP")) {
							// we found a candidate! Get the mention for the tree and add it!
							Mention m2 = treeMentionMap.get(node);
							if (m2 != null) { hobbs.add(m2); }
						}
					}
				}
				
				// 8 If X is an S node, traverse branches of X to the right of P. But stop upon finding any NP or S
				if (xTree.getLabel().equals("S")) {
					boolean seenP = false;
					for (Tree<String> child : xTree.getChildren()) {
						if (!seenP) {
							if (child != pPath)
								continue;
							else
								seenP = true;
						}
						
						// get the list of NP's that happened to not be below any NP or S
						List<Tree<String>> candidates = findNPAboveNPorS(child);
						for (Tree<String> candidate : candidates) {
							// get the corresponding mention and add it to the list
							Mention m2 = treeMentionMap.get(candidate);
							if (m2 != null) { hobbs.add(m2); }
						}
						
					}
				}
				
				// 9 go back to 4
			}
			// Completion of 4: check previous sentence(s) (breadth first) for the first NP to propose as antecedent
			for (int i = 0; i < doc.sentences.size(); i++) {
				if (doc.sentences.get(i) == m.sentence && i > 0) {
					// then we have a previous sentence to explore
					
					Tree<String> prevTree = doc.sentences.get(i - 1).parse;
					for (Tree<String> child : prevTree.getPreOrderTraversal()) {
						if (child.getLabel().equals("NP")) {
							// add this to the list and stop!
							Mention m2 = treeMentionMap.get(child);
							if (m2 != null) { hobbs.add(m2); }
							break;
						}
					}
					break;
				}
			}
			
			candidateMentions.put(m, hobbs);
		}
	    
	    return candidateMentions;
	}

	
	public Map<Mention,HashSet<Mention>> getCoreferences(Document doc){

	    Map<Mention, HashSet<Mention>> candidateMentions = hobbsAlgorithm(doc);

	    return candidateMentions;
		
	}
	

}