class Node:
    def __init__(self, data):
        self.data = data
        self.children = []

class PhTree:
    def __init__(self):
        self.head = None

    def add_state(self, parent_data, data):
        new_node = Node(data)
        if self.head is None:
            self.head = Node(parent_data)
        self.head.children.append(new_node)

    def traversal(self, text, word, environment):
        current_node = self.head
        bestscore = float('inf')
        besttext = text
        second_best = None 
        bestword = word

        for child in current_node.children:
            new_text = text.replace(word, child.data)
            score = environment.compute_cost(new_text)
            if score < bestscore:
                second_best = bestword
                bestscore = score
                bestword = child.data
                besttext = new_text

        return [second_best, besttext]  

class Agent:
    def __init__(self, phoneme_table, vocabulary):
        transformed_data = {}

        for key, values in phoneme_table.items():
            for value in values:
                if value not in transformed_data:
                    transformed_data[value] = [key]
                else:
                    transformed_data[value].append(key)

        self.phoneme_table = transformed_data
        self.vocabulary = vocabulary
        self.best_state = None

    def substitutions(self, word, phoneme_tree, generated_words, originalword, L=0, max_depth=2):
        if L >= len(word) or max_depth == 0:
            return
        
        for R in range(L + 1, len(word) + 1):
            segment = word[L:R]
            if len(segment) <= 2:
                if segment in self.phoneme_table:
                    for substitute in self.phoneme_table[segment]:
                        new_word = word[:L] + substitute + word[R:]
                        if new_word not in generated_words:
                            phoneme_tree.add_state(originalword, new_word)  
                            generated_words.add(new_word)
                            self.substitutions(new_word, phoneme_tree, generated_words, originalword, L + len(substitute), max_depth - 1)
        self.substitutions(word, phoneme_tree, generated_words, originalword, L + 1, max_depth)

    def asr_corrector(self, environment):
        self.text = environment.init_state
        self.words = self.text.split()

        bestscore = float('inf')
        besttext = self.text
        second_best = []
        for word in self.words:
            phoneme_tree = PhTree()
            phoneme_tree.add_state(word, word)
            generated_words = set()  
            self.substitutions(word, phoneme_tree, generated_words, word, 0)
            replacements = phoneme_tree.traversal(besttext, word, environment)
            besttext = replacements[1]
            second_best.append(replacements[0])

        possible = besttext.split()
        original_cost = environment.compute_cost(besttext)
        
        for i in range(len(second_best)):
            text1 = besttext.replace(possible[i], second_best[i])
            cost = environment.compute_cost(text1)
            if cost < original_cost:
                besttext = text1
                original_cost = cost

        bestscore=original_cost
        bestvocabtext = besttext
        for vocab_word in self.vocabulary:
            start_added_text = vocab_word + ' ' + besttext
            start_cost = environment.compute_cost(start_added_text)
            if start_cost < bestscore:
                bestvocabtext = start_added_text
                bestscore = start_cost
        for vocab_word in self.vocabulary:
            end_added_text = bestvocabtext + ' ' + vocab_word
            end_cost = environment.compute_cost(end_added_text)
            if end_cost < bestscore:
                bestvocabtext = end_added_text
                bestscore = end_cost

        self.best_state = bestvocabtext
        environment.best_state = bestvocabtext
        cost = environment.compute_cost(environment.best_state)

