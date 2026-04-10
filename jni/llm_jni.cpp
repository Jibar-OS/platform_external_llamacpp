/*
 * Copyright (C) 2024 The AAOSP Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * JNI bridge between LlmManagerService and llama.cpp.
 *
 * This file is the canonical source. The frameworks/base copy at
 * services/core/jni/com_android_server_llm_LlmManagerService.cpp
 * should be a symlink or copy of this file.
 *
 * Build: Android.bp builds this into libllm_jni.so which is loaded
 * by LlmManagerService via System.loadLibrary("llm_jni").
 */

#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>

#include "llama.h"
#include "common.h"

#define LOG_TAG "LlmJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

struct LlmModel {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    int context_size = 0;
    int n_threads = 4;
};

// Cache JNI method IDs for the callback (avoid repeated lookups)
static jmethodID sOnTokenMethod = nullptr;
static jmethodID sIsCancelledMethod = nullptr;

static void ensureCallbackMethods(JNIEnv* env, jobject callback) {
    if (sOnTokenMethod == nullptr) {
        jclass cls = env->GetObjectClass(callback);
        sOnTokenMethod = env->GetMethodID(cls, "onToken",
                "(Ljava/lang/String;)V");
        sIsCancelledMethod = env->GetMethodID(cls, "isCancelled", "()Z");
        env->DeleteLocalRef(cls);
    }
}

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_android_server_llm_LlmManagerService_nativeLoadModel(
        JNIEnv* env, jclass clazz,
        jstring modelPath, jint contextSize, jint gpuLayers) {

    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    LOGI("Loading model from %s (ctx=%d, gpu_layers=%d)",
         path, contextSize, gpuLayers);

    llama_backend_init();

    // Model params
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = gpuLayers;

    llama_model* model = llama_load_model_from_file(path, model_params);
    env->ReleaseStringUTFChars(modelPath, path);

    if (!model) {
        LOGE("Failed to load model");
        return 0;
    }

    // Context params — tuned for mobile
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = contextSize;
    ctx_params.n_threads = std::max(2, (int)sysconf(_SC_NPROCESSORS_ONLN) / 2);
    ctx_params.n_threads_batch = ctx_params.n_threads;

    llama_context* ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        LOGE("Failed to create context");
        llama_free_model(model);
        return 0;
    }

    LlmModel* llm = new LlmModel();
    llm->model = model;
    llm->ctx = ctx;
    llm->context_size = contextSize;
    llm->n_threads = ctx_params.n_threads;

    LOGI("Model loaded: ctx=%d, threads=%d", contextSize, llm->n_threads);
    return reinterpret_cast<jlong>(llm);
}

JNIEXPORT jstring JNICALL
Java_com_android_server_llm_LlmManagerService_nativeGenerate(
        JNIEnv* env, jclass clazz,
        jlong modelPtr, jstring prompt, jint maxTokens,
        jfloat temperature, jobject callback) {

    LlmModel* llm = reinterpret_cast<LlmModel*>(modelPtr);
    if (!llm || !llm->ctx || !llm->model) {
        LOGE("Invalid model pointer");
        return env->NewStringUTF("");
    }

    ensureCallbackMethods(env, callback);

    const char* promptStr = env->GetStringUTFChars(prompt, nullptr);
    int promptLen = strlen(promptStr);

    // Tokenize
    int maxTokenCount = promptLen + 128; // rough upper bound
    std::vector<llama_token> tokens(maxTokenCount);
    int n_tokens = llama_tokenize(
            llm->model, promptStr, promptLen,
            tokens.data(), tokens.size(),
            /* add_special */ true,
            /* parse_special */ true);
    env->ReleaseStringUTFChars(prompt, promptStr);

    if (n_tokens < 0) {
        // Buffer too small, retry with exact size
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(
                llm->model, promptStr, promptLen,
                tokens.data(), tokens.size(), true, true);
        if (n_tokens < 0) {
            LOGE("Tokenization failed");
            return env->NewStringUTF("");
        }
    }
    tokens.resize(n_tokens);

    if (n_tokens >= llm->context_size) {
        LOGW("Prompt (%d tokens) exceeds context (%d), truncating",
             n_tokens, llm->context_size);
        tokens.resize(llm->context_size - maxTokens - 1);
        n_tokens = tokens.size();
    }

    LOGI("Prompt: %d tokens, generating up to %d tokens (temp=%.2f)",
         n_tokens, maxTokens, temperature);

    // Evaluate prompt in batches
    llama_batch batch = llama_batch_init(512, 0, 1);
    int n_eval = 0;

    while (n_eval < n_tokens) {
        int batch_size = std::min(512, n_tokens - n_eval);
        llama_batch_clear(batch);

        for (int i = 0; i < batch_size; i++) {
            llama_batch_add(batch, tokens[n_eval + i], n_eval + i, {0},
                    /* logits */ (n_eval + i == n_tokens - 1));
        }

        if (llama_decode(llm->ctx, batch) != 0) {
            LOGE("Failed to evaluate prompt batch at position %d", n_eval);
            llama_batch_free(batch);
            llama_kv_cache_clear(llm->ctx);
            return env->NewStringUTF("");
        }

        n_eval += batch_size;
    }

    // Generate tokens
    std::string generated;
    int n_cur = n_tokens;

    // Simple temperature sampling
    auto sample_token = [&]() -> llama_token {
        auto* logits = llama_get_logits_ith(llm->ctx, -1);
        int n_vocab = llama_n_vocab(llm->model);

        std::vector<llama_token_data> candidates(n_vocab);
        for (int i = 0; i < n_vocab; i++) {
            candidates[i] = {i, logits[i], 0.0f};
        }
        llama_token_data_array candidates_p = {
                candidates.data(), (size_t)n_vocab, false};

        if (temperature <= 0.0f) {
            // Greedy
            return llama_sample_token_greedy(llm->ctx, &candidates_p);
        } else {
            llama_sample_temp(llm->ctx, &candidates_p, temperature);
            llama_sample_top_p(llm->ctx, &candidates_p, 0.9f, 1);
            return llama_sample_token(llm->ctx, &candidates_p);
        }
    };

    for (int i = 0; i < maxTokens; i++) {
        // Check cancellation
        if (env->CallBooleanMethod(callback, sIsCancelledMethod)) {
            LOGI("Generation cancelled at token %d", i);
            break;
        }

        llama_token new_token = sample_token();

        // Check for end of generation
        if (llama_token_is_eog(llm->model, new_token)) {
            LOGI("EOS at token %d", i);
            break;
        }

        // Convert token to text
        char buf[256];
        int n = llama_token_to_piece(
                llm->model, new_token, buf, sizeof(buf),
                /* special */ 0, /* parse_special */ false);
        if (n > 0) {
            std::string piece(buf, n);
            generated += piece;

            // Stream to Java callback
            jstring jToken = env->NewStringUTF(piece.c_str());
            env->CallVoidMethod(callback, sOnTokenMethod, jToken);
            env->DeleteLocalRef(jToken);

            // Check for exception (callback might throw RemoteException)
            if (env->ExceptionCheck()) {
                env->ExceptionClear();
                LOGW("Callback exception, stopping generation");
                break;
            }
        }

        // Evaluate the new token
        llama_batch_clear(batch);
        llama_batch_add(batch, new_token, n_cur, {0}, true);
        n_cur++;

        if (llama_decode(llm->ctx, batch) != 0) {
            LOGE("Failed to evaluate generated token %d", i);
            break;
        }
    }

    llama_batch_free(batch);
    llama_kv_cache_clear(llm->ctx);

    LOGI("Generated %zu chars (%d tokens)", generated.size(),
         n_cur - n_tokens);

    return env->NewStringUTF(generated.c_str());
}

JNIEXPORT jstring JNICALL
Java_com_android_server_llm_LlmManagerService_nativeGetModelInfo(
        JNIEnv* env, jclass clazz, jlong modelPtr) {

    LlmModel* llm = reinterpret_cast<LlmModel*>(modelPtr);
    if (!llm || !llm->model) {
        return env->NewStringUTF("No model loaded");
    }

    char buf[256];
    int n = llama_model_meta_val_str(llm->model, "general.name", buf, sizeof(buf));

    std::string info;
    if (n > 0) {
        info = std::string(buf, n);
    } else {
        info = "Unknown model";
    }

    info += " | ctx=" + std::to_string(llm->context_size);
    info += " | threads=" + std::to_string(llm->n_threads);
    info += " | vocab=" + std::to_string(llama_n_vocab(llm->model));

    return env->NewStringUTF(info.c_str());
}

JNIEXPORT void JNICALL
Java_com_android_server_llm_LlmManagerService_nativeUnloadModel(
        JNIEnv* env, jclass clazz, jlong modelPtr) {

    LlmModel* llm = reinterpret_cast<LlmModel*>(modelPtr);
    if (!llm) return;

    LOGI("Unloading model...");
    if (llm->ctx) llama_free(llm->ctx);
    if (llm->model) llama_free_model(llm->model);
    delete llm;

    llama_backend_free();
    LOGI("Model unloaded");
}

} // extern "C"
