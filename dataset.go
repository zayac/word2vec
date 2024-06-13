package main

import (
	"bufio"
	"os"
	"path/filepath"
	"strings"

	"gonum.org/v1/gonum/mat"
)

const (
	sentencesFile = "datasetSentences.txt"
)

// Dataset represents a dataset used in the word2vec application.
type Dataset struct {
	Tokens   []string       // Tokens stores the list of all tokens in the dataset.
	WordToID map[string]int // WordToID maps each unique word to its corresponding ID.
	IDToWord []string       // IDToWord maps each ID to its corresponding word.
}

// newDataset creates a new Dataset object by processing the dataset at the specified path.
func newDataset(path string) (Dataset, error) {
	ds := Dataset{}
	var err error
	if ds.Tokens, err = tokens(path); err != nil {
		return Dataset{}, nil
	}
	ds.WordToID, ds.IDToWord = wordToID(ds.Tokens)
	return ds, nil
}

// tokens reads the dataset file and returns a list of all tokens.
func tokens(path string) ([]string, error) {
	var tokens []string
	f, err := os.Open(filepath.Join(path, sentencesFile))
	if err != nil {
		return nil, err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Split(bufio.ScanWords)
	// Read the remaining lines.
	for scanner.Scan() {
		token := strings.ToLower(scanner.Text())
		tokens = append(tokens, token)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return tokens, nil
}

// wordToID creates a mapping from words to their corresponding IDs.
func wordToID(tokens []string) (map[string]int, []string) {
	wordToID := make(map[string]int)
	var idToWord []string
	var idx int
	for _, token := range tokens {
		if _, ok := wordToID[token]; !ok {
			wordToID[token] = idx
			idToWord = append(idToWord, token)
			idx += 1
		}
	}
	return wordToID, idToWord
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func (ds Dataset) trainingData(window int) (x *mat.Dense, y *mat.Dense) {
	vocabSize := len(ds.IDToWord)
	var pairsCnt int
	var xSl, ySl []float64
	for i, word := range ds.Tokens {
		for idx := max(0, i-window); idx < min(len(ds.Tokens), i+window+1); idx++ {
			if idx == i {
				continue
			}
			xSl = append(xSl, hotVec(ds.WordToID[word], vocabSize)...)
			ySl = append(ySl, hotVec(ds.WordToID[ds.Tokens[idx]], vocabSize)...)
			pairsCnt += 1
		}
	}
	x = mat.NewDense(pairsCnt, vocabSize, xSl)
	y = mat.NewDense(pairsCnt, vocabSize, ySl)
	return
}

// hotVec returns a one-hot vector representation of the given ID.
func hotVec(id int, size int) []float64 {
	v := make([]float64, size)
	v[id] = 1
	return v
}
