#!/bin/bash

# ModelScope 网络诊断脚本
# 用于排查连接问题
#
# 使用方法:
#   bash run_network_diagnose.sh
#
# 诊断项:
#   - 环境变量
#   - DNS 解析
#   - 网络连通性
#   - HTTP 连接
#   - SSL 证书
#
# 参考: reference/wiki.md - 故障排查章节

echo "========================================"
echo "ModelScope 网络诊断"
echo "========================================"
echo "诊断时间: $(date)"
echo ""

# 1. 环境变量
echo "【1. 环境变量】"
echo "  HTTP_PROXY:  ${HTTP_PROXY:-未设置}"
echo "  HTTPS_PROXY: ${HTTPS_PROXY:-未设置}"
echo "  http_proxy:  ${http_proxy:-未设置}"
echo "  https_proxy: ${https_proxy:-未设置}"
echo "  NO_PROXY:    ${NO_PROXY:-未设置}"
echo "  PYTHONPATH:  ${PYTHONPATH:-未设置}"

# 2. DNS 解析
echo ""
echo "【2. DNS 解析】"
echo -n "  www.modelscope.cn: "
if command -v host &> /dev/null; then
    DNS_RESULT=$(host www.modelscope.cn 2>&1)
    if echo "$DNS_RESULT" | grep -q "has address"; then
        echo "$DNS_RESULT" | grep "has address" | head -1 | awk '{print $4}'
    else
        echo "❌ 解析失败"
        echo "     $DNS_RESULT"
    fi
elif command -v nslookup &> /dev/null; then
    nslookup www.modelscope.cn 2>&1 | grep -A 2 "Name:" | tail -1
else
    echo "⚠️ 无 DNS 工具"
fi

# 3. 网络连通性
echo ""
echo "【3. 网络连通性】"

# Ping 测试
echo -n "  Ping ( ICMP ): "
PING_RESULT=$(ping -c 2 -W 3 www.modelscope.cn 2>&1)
if [ $? -eq 0 ]; then
    echo "$PING_RESULT" | tail -1 | awk -F'/' '{print "✅ 平均延迟: " $5 "ms"}'
else
    echo "❌ 不可达 (可能被防火墙拦截)"
fi

# TCP 连接测试
echo -n "  TCP 443 端口: "
if command -v nc &> /dev/null; then
    if timeout 5 nc -zv www.modelscope.cn 443 2>&1 | grep -q "succeeded"; then
        echo "✅ 端口开放"
    else
        echo "❌ 端口关闭或超时"
    fi
elif command -v timeout &> /dev/null; then
    if timeout 5 bash -c "echo > /dev/tcp/www.modelscope.cn/443" 2>/dev/null; then
        echo "✅ 端口开放"
    else
        echo "❌ 端口关闭或超时"
    fi
else
    echo "⚠️ 无法测试 TCP 端口"
fi

# 4. HTTP 连接
echo ""
echo "【4. HTTP 连接测试】"

# 不使用代理
echo -n "  直连 (无代理): "
DIRECT_CODE=$(timeout 10 curl -k -s -o /dev/null -w "%{http_code}" 'https://www.modelscope.cn' 2>/dev/null || echo "000")
if [ "$DIRECT_CODE" = "200" ] || [ "$DIRECT_CODE" = "301" ] || [ "$DIRECT_CODE" = "302" ]; then
    echo "✅ HTTP $DIRECT_CODE"
else
    echo "❌ HTTP $DIRECT_CODE (连接失败)"
fi

# 使用代理
if [ -n "$HTTPS_PROXY" ] || [ -n "$https_proxy" ]; then
    PROXY_ADDR="${HTTPS_PROXY:-$https_proxy}"
    # 截断代理地址显示，避免泄露凭据
    PROXY_DISPLAY=$(echo "$PROXY_ADDR" | sed 's|//[^@]*@|//***@|' | cut -c1-60)
    echo -n "  代理连接 ($PROXY_DISPLAY): "
    PROXY_CODE=$(timeout 10 curl -k -s -o /dev/null -w "%{http_code}" --proxy "$PROXY_ADDR" 'https://www.modelscope.cn' 2>/dev/null || echo "000")
    if [ "$PROXY_CODE" = "200" ] || [ "$PROXY_CODE" = "301" ] || [ "$PROXY_CODE" = "302" ]; then
        echo "✅ HTTP $PROXY_CODE"
    else
        echo "❌ HTTP $PROXY_CODE (代理可能配置错误)"
    fi
fi

# 响应时间
echo -n "  响应时间: "
RESPONSE_TIME=$(timeout 15 curl -k -s -o /dev/null -w "%{time_total}" 'https://www.modelscope.cn' 2>/dev/null || echo "timeout")
if [ "$RESPONSE_TIME" != "timeout" ]; then
    echo "${RESPONSE_TIME}s"
else
    echo "超时 (>15s)"
fi

# 5. SSL 证书
echo ""
echo "【5. SSL 证书信息】"
if command -v openssl &> /dev/null; then
    CERT_INFO=$(timeout 10 openssl s_client -connect www.modelscope.cn:443 -servername www.modelscope.cn 2>/dev/null <<EOF
QUIT
EOF
)
    
    if [ -n "$CERT_INFO" ]; then
        # 提取颁发者
        ISSUER=$(echo "$CERT_INFO" | openssl x509 -issuer -noout 2>/dev/null | sed 's/issuer=//')
        echo "  颁发者: $ISSUER"
        
        # 提取有效期
        EXPIRY=$(echo "$CERT_INFO" | openssl x509 -dates -noout 2>/dev/null | grep "notAfter" | sed 's/notAfter=//')
        echo "  有效期至: $EXPIRY"
        
        # 验证证书
        VERIFY_RESULT=$(echo "$CERT_INFO" | grep "Verify return code" | awk -F':' '{print $2}' | xargs)
        echo "  验证结果: $VERIFY_RESULT"
    else
        echo "  ❌ SSL 握手失败"
    fi
else
    echo "  ⚠️ openssl 未安装，无法检查证书"
fi

# 6. Python SSL 测试
echo ""
echo "【6. Python SSL 测试】"
python3 -c "
import ssl, socket, sys

print('  Python SSL 版本:', ssl.OPENSSL_VERSION)

# 测试 1: 默认验证
try:
    ctx = ssl.create_default_context()
    with socket.create_connection(('www.modelscope.cn', 443), timeout=10) as s:
        with ctx.wrap_socket(s, server_hostname='www.modelscope.cn'):
            pass
    print('  ✅ 默认验证: 成功')
except ssl.SSLCertVerificationError as e:
    print('  ❌ 默认验证: 失败 (证书验证错误)')
    print('     原因:', str(e)[:60])
except Exception as e:
    print('  ❌ 默认验证: 失败')
    print('     原因:', str(e)[:60])

# 测试 2: 禁用验证
try:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with socket.create_connection(('www.modelscope.cn', 443), timeout=10) as s:
        with ctx.wrap_socket(s, server_hostname='www.modelscope.cn'):
            pass
    print('  ✅ 禁用验证: 成功 (可使用此方式绕过证书问题)')
except Exception as e:
    print('  ❌ 禁用验证: 失败 -', str(e)[:60])
" 2>&1

# 7. 诊断建议
echo ""
echo "========================================"
echo "诊断建议"
echo "========================================"
echo ""

SUGGESTIONS=()

# 检查代理
if [ -z "$HTTPS_PROXY" ] && [ -z "$https_proxy" ] && [ "$DIRECT_CODE" != "200" ]; then
    SUGGESTIONS+=("1. 配置代理: export HTTPS_PROXY=http://proxy-host:port/")
    SUGGESTIONS+=("   或使用代理配置脚本: bash scripts/setup_proxy.sh")
fi

# 检查 SSL
if [ "$VERIFY_RESULT" != "0 (ok)" ] 2>/dev/null; then
    SUGGESTIONS+=("2. SSL 证书验证失败，可尝试：")
    SUGGESTIONS+=("   - 安装可信 CA 证书到系统")
    SUGGESTIONS+=("   - 禁用验证: export PYTHONSSLVERIFY=0")
fi

# 检查网络
if [ "$DIRECT_CODE" = "000" ]; then
    SUGGESTIONS+=("3. 网络不通，检查防火墙或联系网络管理员")
fi

# 下载线程优化
SUGGESTIONS+=("4. 下载速度优化: export MODELSCOPE_DOWNLOAD_THREAD_NUM=8")

if [ ${#SUGGESTIONS[@]} -eq 0 ]; then
    echo "✅ 未发现问题，网络配置正常"
else
    echo "⚠️  发现问题或优化建议："
    for suggestion in "${SUGGESTIONS[@]}"; do
        echo "   $suggestion"
    done
fi

echo ""
echo "========================================"
echo ""
echo "📖 更多故障排查方法，参考: reference/wiki.md - 故障排查章节"
