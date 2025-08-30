from typing import Optional

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Flatten, Conv1D, Bidirectional, GlobalAveragePooling1D, \
    RepeatVector, Subtract, TimeDistributed, Activation, LeakyReLU, GaussianNoise, Dense, LayerNormalization, Add, \
    Concatenate, Lambda, Multiply, Reshape, Dropout, MultiHeadAttention
from keras.src.regularizers import L2
from tensorflow.keras.layers import MultiHeadAttention

from CloudScripts.functions import PositionalEncoding, transformer_block, RecentTrendExtractor, CLSToken, \
    fusion_softmax_fn, fusion_token_fn, stack_multi_context_fn, attention_softmax_fn, attention_pool_fn, Sign
import tensorflow as tf

def create_market_momentum_model(seq_length, n_features, n_sectors, embedding_dim_sector, future_days, include_confidence):
    ################################################################################
    ###                               INPUTS                                     ###
    ################################################################################
    #sector_input = Input(shape=(1,), name="sector_input", dtype='int32')
    data_input = Input(shape=(seq_length, n_features), name="data_input")
    projected_features = TimeDistributed(Dense(16, activation='linear'))(data_input)

    ################################################################################
    ###                POSITIONAL ENCODING & SECTOR EMBEDDING                    ###
    ################################################################################
    # Apply positional encoding
    data_with_pos = PositionalEncoding(seq_length, 16)(projected_features)
    raw_data_pos = PositionalEncoding(seq_length, n_features)(data_input)
    ################################################################################
    ###                  MARKET CONTEXT BLOCK (CONV + TRANSFORMER)             ###
    ################################################################################
    # Market-wide pattern recognition via convolution
    market_conv = Conv1D(64, kernel_size=5, padding='same', activation='relu', kernel_regularizer=L2(1e-4), name="market_convolution")(data_with_pos)
    market_conv = LayerNormalization()(market_conv)

    # Cross-market attention using a transformer block
    market_transformer = transformer_block(data_with_pos, num_heads=4, ff_dim=128, dropout_rate=0.2)
    market_transformer = Dense(32, name="transformer_dense")(market_transformer)
    market_conv_proj = Dense(32)(market_conv)
    market_fused = Add()([market_conv_proj, market_transformer])
    market_attention = LayerNormalization()(market_fused)

    # Full sequence LSTM with market context
    main_lstm = Bidirectional(LSTM(128, return_sequences=True, name="bi_main_lstm"))(market_attention)
    main_lstm = LSTM(32, return_sequences=False, recurrent_dropout=0.1, name="main_lstm")(main_lstm)
    main_lstm = LayerNormalization()(main_lstm)
    main_lstm = Dropout(0.2)(main_lstm)

    ################################################################################
    ###                       MOMENTUM TREND FEATURES                           ###
    ################################################################################
    # Recent trend (short-term)
    recent_trend = RecentTrendExtractor(steps=5)(market_attention)
    recent_conv = Conv1D(32, kernel_size=3, padding='same', kernel_regularizer=L2(1e-5), name="recent_conv")(recent_trend)
    recent_conv = LayerNormalization()(recent_conv)
    recent_pool = GlobalAveragePooling1D()(recent_conv)

    # Medium trend (medium-term)
    medium_trend = RecentTrendExtractor(steps=20)(market_attention)
    medium_conv = Conv1D(32, kernel_size=5, padding='same', kernel_regularizer=L2(1e-5), name="medium_conv")(medium_trend)
    medium_conv = LayerNormalization()(medium_conv)
    medium_pool = GlobalAveragePooling1D()(medium_conv)

    ################################################################################
    ###                          PRICE PATTERN EXTRACTOR                        ###
    ################################################################################
    # Extract raw price channels
    raw_price_input = Lambda(
        lambda x: x[..., :3],
        name="raw_price_input",
        output_shape=lambda input_shape: input_shape[:-1] + (3,)
    )(raw_data_pos)
    raw_price_input = LayerNormalization()(raw_price_input)

    recent_data = RecentTrendExtractor(steps=5)(raw_price_input)
    conv_short_3 = Conv1D(32, 3, padding='same')(recent_data)
    conv_short_5 = Conv1D(32, 5, padding='same')(recent_data)
    recent_conv_data = Concatenate()([conv_short_3, conv_short_5])

    medium_data = RecentTrendExtractor(steps=20)(raw_price_input)
    medium_conv_data = Conv1D(16, kernel_size=5, padding='same', name="medium_conv_data")(medium_data)

    # LSTM encoding of both timeframes
    recent_lstm = LSTM(32, return_sequences=True, name="recent_lstm")(recent_conv_data)
    medium_lstm = LSTM(32, return_sequences=True, name="medium_lstm")(medium_conv_data)

    # Create position indices (static tensors)
    recent_pos_indices = tf.range(start=0, limit=5, delta=1)
    medium_pos_indices = tf.range(start=0, limit=20, delta=1)

    # Learnable embeddings
    pos_embedding = Embedding(input_dim=50, output_dim=32, name="positional_embedding")

    # Positional embedding added to each timestep
    pos_encoding_recent = pos_embedding(recent_pos_indices)
    pos_encoding_recent = tf.expand_dims(pos_encoding_recent, axis=0)
    recent_pos_encoded = Add(name="recent_pos_add")([recent_lstm, pos_encoding_recent])

    pos_encoding_medium = pos_embedding(medium_pos_indices)
    pos_encoding_medium = tf.expand_dims(pos_encoding_medium, axis=0)
    medium_pos_encoded = Add(name="medium_pos_add")([medium_lstm, pos_encoding_medium])

    # Concatenate along time axis
    combined_lstm_features = Concatenate(axis=1, name="concat_lstm_temporal")([recent_pos_encoded, medium_pos_encoded])

    # LSTM to combine all price features
    fusion_seq = LSTM(32, return_sequences=True, recurrent_dropout=0.05, name="fusion_seq")(combined_lstm_features)
    fusion_dense = Dense(32, name="fusion_dense")(fusion_seq)
    fusion_dense = LayerNormalization()(fusion_dense)
    fusion_dense = Dropout(0.1)(fusion_dense)

    # Learnable attention pooling
    attn_scores = Dense(1, name="fusion_attn_scores")(fusion_dense)
    attn_scores = Lambda(
        fusion_softmax_fn,
        name="fusion_softmax",
        output_shape=lambda input_shape: input_shape
    )(attn_scores)

    weighted = Multiply(name="fusion_weighted_seq")([fusion_dense, attn_scores])
    fusion_token = Lambda(
        fusion_token_fn,
        name="fusion_token",
        output_shape=lambda input_shape: (input_shape[0], input_shape[-1])
    )(weighted)


    ################################################################################
    ###                         TECHNICAL FEATURES                              ###
    ################################################################################
    # Technical features processing
    technical_input = Lambda(
        lambda x: x[..., 2:],
        name="technical_input",
        output_shape=lambda input_shape: input_shape[:-1] + (input_shape[-1] - 2,)
    )(raw_data_pos)
    technical_input = LayerNormalization()(technical_input)

    # Short and medium-term convolution
    tech_short = RecentTrendExtractor(steps=5)(technical_input)
    tech_medium = RecentTrendExtractor(steps=20)(technical_input)
    conv_short_tech = Conv1D(16, 3, padding='same')(tech_short)
    conv_medium_tech = Conv1D(16, 5, padding='same')(tech_medium)

    # Pool and encode with LSTM
    short_pool_tech = GlobalAveragePooling1D()(conv_short_tech)
    medium_pool_tech = GlobalAveragePooling1D()(conv_medium_tech)

    tech_combined = Concatenate()([short_pool_tech, medium_pool_tech])
    tech_combined = Reshape((1, 32))(tech_combined)
    tech_lstm = LSTM(32, return_sequences=True, name="tech_lstm")(tech_combined)

    # Cross-attention: technical over price features
    cross_attention = MultiHeadAttention(num_heads=4, key_dim=32, name="cross_attention_tech_on_price")(
        query=tech_lstm,
        key=fusion_seq,
        value=fusion_seq,
    )
    cross_attention = LayerNormalization(name="cross_attention_norm")(cross_attention)

    technical_pool = GlobalAveragePooling1D()(cross_attention)
    technical_pool = Dense(32, activation='relu', name="technical_pool")(technical_pool)

    ################################################################################
    ###                      FINAL MULTI-CONTEXT ATTENTION                      ###
    ################################################################################

    context = Concatenate(axis=1, name='context_tokens')([
        Reshape((1, 32))(fusion_token),
        Reshape((1, 32))(technical_pool),
    ])

    query = Reshape((1, 32))(main_lstm)

    attn = MultiHeadAttention(num_heads=4, key_dim=32)(query=query, value=context, key=context)
    attn_scores = Dense(1, name="attention_scores")(attn)
    # 2. Apply softmax over time axis
    attn_scores = Lambda(
        attention_softmax_fn,
        name="attention_softmax",
        output_shape=lambda input_shape: input_shape
    )(attn_scores)
    weighted_attn = Multiply(name="attention_weighted")([attn, attn_scores])
    attention_pool = Lambda(
        attention_pool_fn,
        name="attention_pool",
        output_shape=lambda input_shape: (input_shape[0], input_shape[-1])
    )(weighted_attn)

    combined = Concatenate()([main_lstm, attention_pool])
    attention_pool = Dense(32, activation='relu')(combined)

    ################################################################################
    ###                          DENSE HEAD + OUTPUT                            ###
    ################################################################################
    # Combine all features and apply dropout
    sub_vecs = [recent_pool, medium_pool, attention_pool, fusion_token, technical_pool]
    gated_vecs = []

    for i, vec in enumerate(sub_vecs):
        gate_dense = Dense(vec.shape[-1], activation='sigmoid', name=f'gate_block_{i}')
        gate = gate_dense(vec)
        gated_vecs.append(Multiply()([vec, gate]))

    fused_inputs = Concatenate()(gated_vecs)

    res_fused = Add()([fused_inputs, Concatenate()(sub_vecs)])
    all_features = LayerNormalization()(res_fused)
    all_features = Dropout(0.2)(all_features)

    # Dense prediction layers
    dense1_proj = Dense(128, kernel_regularizer=L2(1e-5), activation='linear')(all_features)
    dense1 = Dense(128, kernel_regularizer=L2(1e-5), activation='gelu')(all_features)
    dense1 = Add()([dense1, dense1_proj])  # residual
    dense1 = LayerNormalization()(dense1)
    dense1 = Dropout(0.3)(dense1)

    dense2 = Dense(32, kernel_regularizer=L2(1e-5), activation='gelu')(dense1)
    dense2 = LayerNormalization()(dense2)
    dense2 = Dropout(0.1)(dense2)

    # Feature-wise gating
    gate = Dense(32, activation='sigmoid')(dense2)
    transform = Dense(32, activation='relu')(dense2)
    dense2 = Add()([gate * transform, (1 - gate) * dense2])
    dense2 = LayerNormalization()(dense2)

    bottleneck = Dense(16, activation='relu', kernel_regularizer=L2(1e-5), name="bottleneck")(dense2)
    bottleneck = Dropout(0.05)(bottleneck)

    output_mu = Dense(future_days, activation='linear', name="output_mu", kernel_regularizer=L2(1e-5))(bottleneck)

    if include_confidence:                                      #  joint training
        confidence      = Dense(1, activation='sigmoid', name="confidence", kernel_regularizer=L2(1e-5))(bottleneck)
        final_output    = Concatenate(name="final_distribution_output")([output_mu, confidence])
    else:                                                       # stage-1 (price only)
        final_output    = output_mu


    return Model(inputs=[data_input], outputs=final_output)




