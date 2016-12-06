{-# LANGUAGE PackageImports #-}

module HNeuron
    ( predict
    , train
    , trainingSetFromPairs
    , sigmoid
    , mkRandomSynapses
    , inputLayer
    , hiddenLayer
    , outputLayer
    ) where

import           "base" Data.Foldable (foldl', sum)
import           "matrix" Data.Matrix (Matrix)
import qualified "matrix" Data.Matrix as M
import qualified "random" System.Random as R
import qualified "MonadRandom" Control.Monad.Random as R

trainingSetFromPairs :: [([Float], [Float])] -> [(Matrix Activation, Matrix Activation)]
trainingSetFromPairs = map $ \(inp, out) -> ( M.fromList 1 (length inp) inp
                                           , M.fromList 1 (length out) out
                                           )

newtype NeuralBuilder = Layer Int

inputLayer :: Int -> NeuralBuilder
inputLayer = Layer

hiddenLayer :: Int -> NeuralBuilder
hiddenLayer = Layer

outputLayer :: Int -> NeuralBuilder
outputLayer = Layer

mkSynapses :: [NeuralBuilder]   -- layer list
           -> [Matrix Weight]   -- synapses
mkSynapses builder = [ M.fromList 2 3 [-0.7, 1.0, 0.5, 0, -0.2, 0.1]
                   , M.fromList 3 1 [1.0, 0.0, -0.5]
                   ]

mkRandomSynapses :: R.RandomGen g
                 => [NeuralBuilder]           -- layer list
                 -> R.Rand g [Matrix Weight]  -- synapses
mkRandomSynapses builder =
  mapM randomSynapseFromBuilders $ zip builder $ tail builder


randomSynapseFromBuilders :: R.RandomGen g
                          => (NeuralBuilder, NeuralBuilder) -- adjacent layers
                          -> R.Rand g (Matrix Weight)       -- random initialized synapse
randomSynapseFromBuilders (Layer x, Layer y) =
  sequence $ M.fromList x y $ replicate (x*y) randomWeight

randomWeight :: (R.MonadRandom m) => m Float
randomWeight = R.getRandomR (-1.0, 1.0)

predict :: (Value -> Activation)  -- activation function
        -> [Matrix Weight]        -- synapses
        -> Matrix Activation      -- input vector
        -> Matrix Activation      -- output vector
predict act synapses input = last $ forwardProp act synapses input


train :: [ ( Matrix Activation  -- input
           , Matrix Activation  -- expected output
           )
         ]                      -- training set
      -> [Matrix Weight]        -- synapses
      -> [Matrix Weight]        -- trained synapses
train [] syns = syns
train ((input,expected):trainingSet) syns = train trainingSet newSyns
  where
    newSyns = reverse newSynsR
    newSynsR = applyDeltas activationsSynapsesDeltas
    activationsSynapsesDeltas = map (\((layer, syn), delta) -> (layer,syn,delta)) $ zip activationsSynsR deltas
    deltas = deltaOutput:innerDeltas
    innerDeltas = mapWithAcc (\dx1 (lx,syn) -> delta lx syn dx1) deltaOutput activationsSynsR 
    deltaOutput = (expected - outputLayerActivation) `elemwiseProd` fmap dSigmoid outputLayerActivation
    outputLayerActivation = last activations
    activationsSynsR = reverse $ zip activations syns
    activations = input : forwardProp sigmoid syns input


applyDeltas :: [ ( Matrix Activation  -- activation layer x
                 , Matrix Weight      -- synapse    layers x -> x+1
                 , Matrix Delta       -- delta      layer x+1
                 )
               ] -- in reverse order
            -> [Matrix Weight] -- synapses w/ applied deltas in reverse order
applyDeltas = map applyDelta
  where applyDelta (layer,synapse,delta) = synapse + M.transpose layer * delta


delta :: Matrix Activation  -- layer x activations
      -> Matrix Weight      -- synapses (x | x+1)
      -> Matrix Delta       -- delta of layer x+1
      -> Matrix Delta       -- delta of layer x
delta layerA synAB deltaLayerB =
  deltaLayerB *  M.transpose synAB `elemwiseProd` fmap dSigmoid layerA


forwardProp :: (Value -> Float)     -- activation function
            -> [Matrix Weight]      -- synapses
            -> Matrix Activation    -- input row vector
            -> [Matrix Activation]  -- layers activations
forwardProp activation synapses input = -- output layers activations
  mapWithAcc (\x y -> activation <$> (x * y)) input synapses


--------- UTIL --------------

mapWithAcc :: (b -> a -> b) -> b -> [a] -> [b]
mapWithAcc _ _ [] = []
mapWithAcc f x xs = fst $ foldl' (iterfunc f) ([],x) xs
  where
    iterfunc f (lst, val) x = (lst ++ [f val x], f val x)

elemwiseProd :: (Num a) => Matrix a -> Matrix a -> Matrix a
elemwiseProd = M.elementwise (*)

--------- MODEL -------------
type Value = Float
type Weight = Value
type Activation = Value
type Delta = Value
type Synapse = Matrix Weight

newtype NeuralNetwork =
  NeuralNetwork
    { connections :: [Synapse]
    }

------------------------------
sigmoid :: Value -> Activation
sigmoid x = 1 / (1 + exp (-x))

dSigmoid :: Activation -> Value
dSigmoid x = x * (1 - x)
