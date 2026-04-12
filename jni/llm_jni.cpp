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
 * JNI bridge between LlmManagerService and llama.cpp (b4547+ API).
 *
 * Canonical location: external/llama.cpp/jni/llm_jni.cpp
 * Built into libllm_jni.so via Android.bp.
 *
 * Uses the modern llama.cpp sampler chain API (post-b3000).
 */

#include <jni.h>
#include <string>
#include <vector>
#include <mutex>
#include <android/log.h>
#include <unistd.h>

#include "llama.h"

// Inline batch helpers (avoiding common.h dependency which pulls in minja.hpp)
static void llama_batch_clear_local(struct llama_batch & batch) {
    batch.n_tokens = 0;
}

static void llama_batch_add_local(
    struct llama_batch & batch,
    llama_token id,
    llama_pos pos,
    const std::vector<llama_seq_id> & seq_ids,
    bool logits) {
    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits  [batch.n_tokens] = logits;
    batch.n_tokens++;
}


#define LOG_TAG "LlmJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

struct LlmModel {
    llama_model* model = nullptr;
    const llama_vocab* vocab = nullptr;
    llama_context* ctx = nullptr;
    int context_size = 0;
    int n_threads = 4;
    std::mutex mutex; // Guards against concurrent use/free
};

// Backend initialized once via JNI_OnLoad
static bool sBackendInitialized = false;

extern "C" {

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
    if (!sBackendInitialized) {
        llama_backend_init();
        sBackendInitialized = true;
        LOGI("llama.cpp backend initialized");
    }
    return JNI_VERSION_1_6;
}

JNIEXPORT void JNICALL JNI_OnUnload(JavaVM* vm, void* reserved) {
    if (sBackendInitialized) {
        llama_backend_free();
        sBackendInitialized = false;
        LOGI("llama.cpp backend freed");
    }
}

JNIEXPORT jlong JNICALL
Java_com_android_server_llm_LlmManagerService_nativeLoadModel(
        JNIEnv* env, jclass clazz,
        jstring modelPath, jint contextSize, jint gpuLayers,
        jint nThreads) {

    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    if (!path) {
        LOGE("Failed to get model path string");
        return 0;
    }

    LOGI("Loading model from %s (ctx=%d, gpu_layers=%d, threads=%d)",
         path, contextSize, gpuLayers, nThreads);

    // Model params
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = gpuLayers;

    llama_model* model = llama_model_load_from_file(path, model_params);
    env->ReleaseStringUTFChars(modelPath, path);

    if (!model) {
        LOGE("Failed to load model");
        return 0;
    }

    // Context params
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = contextSize;
    int threads = (nThreads > 0)
            ? nThreads
            : std::max(2, (int)sysconf(_SC_NPROCESSORS_ONLN) / 2);
    ctx_params.n_threads = threads;
    ctx_params.n_threads_batch = threads;

    llama_context* ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        LOGE("Failed to create context");
        llama_model_free(model);
        return 0;
    }

    LlmModel* llm = new LlmModel();
    llm->model = model;
    llm->vocab = llama_model_get_vocab(model);
    llm->ctx = ctx;
    llm->context_size = contextSize;
    llm->n_threads = threads;

    LOGI("Model loaded: ctx=%d, threads=%d", contextSize, threads);
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

    // Lock the model for this generation
    std::lock_guard<std::mutex> lock(llm->mutex);

    // Get callback method IDs fresh each call (safe across classloaders)
    jclass callbackClass = env->GetObjectClass(callback);
    if (!callbackClass) {
        LOGE("Failed to get callback class");
        return env->NewStringUTF("");
    }
    jmethodID onTokenMethod = env->GetMethodID(
            callbackClass, "onToken", "(Ljava/lang/String;)V");
    jmethodID isCancelledMethod = env->GetMethodID(
            callbackClass, "isCancelled", "()Z");
    env->DeleteLocalRef(callbackClass);

    if (!onTokenMethod || !isCancelledMethod) {
        LOGE("Failed to find callback methods");
        return env->NewStringUTF("");
    }

    // Get prompt string — keep the reference until we're done tokenizing
    const char* promptStr = env->GetStringUTFChars(prompt, nullptr);
    if (!promptStr) {
        LOGE("Failed to get prompt string");
        return env->NewStringUTF("");
    }
    int promptLen = (int)strlen(promptStr);

    // Tokenize — first call to get required size
    int n_tokens = llama_tokenize(
            llm->vocab, promptStr, promptLen,
            nullptr, 0,
            /* add_special */ true,
            /* parse_special */ true);

    // n_tokens is negative, indicating the required buffer size
    if (n_tokens < 0) {
        n_tokens = -n_tokens;
    }

    std::vector<llama_token> tokens(n_tokens);
    n_tokens = llama_tokenize(
            llm->vocab, promptStr, promptLen,
            tokens.data(), tokens.size(),
            /* add_special */ true,
            /* parse_special */ true);

    // Done with prompt string
    env->ReleaseStringUTFChars(prompt, promptStr);

    if (n_tokens < 0) {
        LOGE("Tokenization failed");
        return env->NewStringUTF("");
    }
    tokens.resize(n_tokens);

    // Truncate if prompt exceeds context
    int maxGen = maxTokens > 0 ? maxTokens : 2048;
    if (n_tokens + maxGen > llm->context_size) {
        if (n_tokens >= llm->context_size) {
            int keep = llm->context_size - maxGen - 1;
            if (keep < 1) keep = 1;
            LOGW("Prompt (%d tokens) exceeds context (%d), truncating to %d",
                 n_tokens, llm->context_size, keep);
            tokens.resize(keep);
            n_tokens = keep;
        }
    }

    LOGI("Prompt: %d tokens, generating up to %d tokens (temp=%.2f)",
         n_tokens, maxGen, temperature);

    // Evaluate prompt in batches
    llama_batch batch = llama_batch_init(512, 0, 1);
    int n_eval = 0;

    while (n_eval < n_tokens) {
        int batch_size = std::min(512, n_tokens - n_eval);
        llama_batch_clear_local(batch);

        for (int i = 0; i < batch_size; i++) {
            bool is_last = (n_eval + i == n_tokens - 1);
            llama_batch_add_local(batch, tokens[n_eval + i], n_eval + i,
                    {0}, is_last);
        }

        if (llama_decode(llm->ctx, batch) != 0) {
            LOGE("Failed to evaluate prompt batch at position %d", n_eval);
            llama_batch_free(batch);
            llama_kv_cache_clear(llm->ctx);
            return env->NewStringUTF("");
        }

        n_eval += batch_size;
    }

    // Set up sampler chain (modern llama.cpp API)
    llama_sampler* sampler = llama_sampler_chain_init(
            llama_sampler_chain_default_params());

    if (temperature <= 0.0f) {
        // Greedy sampling
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    } else {
        // Temperature + top-p sampling
        llama_sampler_chain_add(sampler,
                llama_sampler_init_top_p(0.9f, /* min_keep */ 1));
        llama_sampler_chain_add(sampler,
                llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(sampler,
                llama_sampler_init_dist(/* seed */ 0));
    }

    // Generate tokens
    std::string generated;
    int n_cur = n_tokens;

    for (int i = 0; i < maxGen; i++) {
        // Check cancellation
        jboolean cancelled = env->CallBooleanMethod(callback, isCancelledMethod);
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
            LOGW("Exception checking cancellation, stopping");
            break;
        }
        if (cancelled) {
            LOGI("Generation cancelled at token %d", i);
            break;
        }

        // Sample next token
        llama_token new_token = llama_sampler_sample(sampler, llm->ctx, -1);

        // Check for end of generation
        if (llama_token_is_eog(llm->vocab, new_token)) {
            LOGI("EOS at token %d", i);
            break;
        }

        // Accept the token (updates sampler state)
        llama_sampler_accept(sampler, new_token);

        // Convert token to text
        char buf[256];
        int n = llama_token_to_piece(
                llm->vocab, new_token, buf, sizeof(buf),
                /* special */ 0, /* parse_special */ false);
        if (n > 0) {
            std::string piece(buf, n);
            generated += piece;

            // Stream to Java callback
            jstring jToken = env->NewStringUTF(piece.c_str());
            env->CallVoidMethod(callback, onTokenMethod, jToken);
            env->DeleteLocalRef(jToken);

            // Check for exception from callback
            if (env->ExceptionCheck()) {
                env->ExceptionClear();
                LOGW("Callback exception, stopping generation");
                break;
            }
        }

        // Evaluate the new token
        llama_batch_clear_local(batch);
        llama_batch_add_local(batch, new_token, n_cur, {0}, true);
        n_cur++;

        if (llama_decode(llm->ctx, batch) != 0) {
            LOGE("Failed to evaluate generated token %d", i);
            break;
        }
    }

    // Cleanup
    llama_sampler_free(sampler);
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
    int n = llama_model_meta_val_str(llm->model, "general.name",
            buf, sizeof(buf));

    std::string info;
    if (n > 0) {
        info = std::string(buf, n);
    } else {
        info = "Unknown model";
    }

    info += " | ctx=" + std::to_string(llm->context_size);
    info += " | threads=" + std::to_string(llm->n_threads);
    info += " | vocab=" + std::to_string(llama_n_vocab(llm->vocab));

    return env->NewStringUTF(info.c_str());
}

JNIEXPORT void JNICALL
Java_com_android_server_llm_LlmManagerService_nativeUnloadModel(
        JNIEnv* env, jclass clazz, jlong modelPtr) {

    LlmModel* llm = reinterpret_cast<LlmModel*>(modelPtr);
    if (!llm) return;

    // Lock to prevent unloading during generation
    std::lock_guard<std::mutex> lock(llm->mutex);

    LOGI("Unloading model...");
    if (llm->ctx) {
        llama_free(llm->ctx);
        llm->ctx = nullptr;
    }
    if (llm->model) {
        llama_model_free(llm->model);
        llm->model = nullptr;
    }
    delete llm;

    LOGI("Model unloaded");
}

} // extern "C"
