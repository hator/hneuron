{-# LANGUAGE PackageImports #-}

module Main where

import HNeuron (mkRandomSynapses, train, predict, trainingSetFromPairs, sigmoid, inputLayer, hiddenLayer, outputLayer)
import qualified "MonadRandom" Control.Monad.Random as R

-- FIXME
import qualified Data.Matrix as M

main :: IO ()
main = do
  let trainingSet = trainingSetFromPairs -- XOR
        [ ([0,0], [0])
        , ([0,1], [1])
        , ([1,0], [1])
        , ([1,1], [0])
        ]
  neuralNet <- R.evalRandIO $ mkRandomSynapses [ inputLayer 2
                                               , hiddenLayer 3
                                               , outputLayer 1
                                               ]
  let generations = iterate (train trainingSet) neuralNet
  let examples = take 6 $ everynth 10000 generations
  let (inputs, expected) = unzip trainingSet
  let lastGen = last examples
  let predictions = (\syn -> predict sigmoid syn <$> inputs) <$> examples
  let errors = (rms . fmap (uncurry rmsErrMat)) . uncurry zip <$> zip (repeat expected) predictions
  putStrLn $ "RMS Errors of every 10000th generation: " ++ show (errors :: [Float])
  putStrLn $ "Output of trained net: " ++ show (map (predict sigmoid lastGen) inputs)

-- TODO move to Util
everynth :: Int -> [a] -> [a]
everynth n [] = []
everynth n (x:xs) = x : everynth n (drop (n-1) xs)

rms :: (Foldable t, Functor t, Real a, Floating b) => t a -> b
rms xs = sqrt $ squares / len
  where
    squares = realToFrac $ sum $ (^2) <$> xs
    len = realToFrac $ length xs

rmsErr :: (Real a, Floating b) => [a] -> [a] -> b
rmsErr expected actual = rms $ uncurry (-) <$> zip expected actual

rmsErrMat :: (Real a, Floating b) => M.Matrix a -> M.Matrix a -> b
rmsErrMat expected actual = sqrt $ squares / len
  where
    squares = realToFrac $ sum $ (^2) <$> expected - actual
    len = realToFrac $ length expected
